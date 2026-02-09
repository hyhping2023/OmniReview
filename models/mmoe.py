import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class Expert(nn.Module):
    """单个专家网络"""
    def __init__(self, input_dim: int, dropout: float, expert_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, expert_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_size, expert_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class Gate(nn.Module):
    """门控网络"""
    def __init__(self, input_dim: int, num_experts: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class TaskTower(nn.Module):
    """任务塔网络"""
    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MMoE(nn.Module):
    """内容感知MoE：门控由交互特征驱动"""
    def __init__(self, emb_dim: int, num_experts: int, dropout: float, expert_size:int, 
                 num_tasks: int, router_noise: float = 0.1, topk:int = 3):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.dropout = dropout
        self.router_noise = router_noise
        self.num_tasks = num_tasks
        self.topk = topk
        
        # 1. 交互特征提取层
        self.interaction_encoder = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim * 4),
            nn.LayerNorm(self.emb_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.emb_dim * 4, self.emb_dim * 2),
            nn.LayerNorm(self.emb_dim * 2)
        )
        
        # 2. 门控网络（输入=交互特征）
        self.multi_gating_network = nn.ModuleList([
            Gate(self.emb_dim * 2, self.num_experts, self.dropout) for _ in range(self.num_tasks)
        ])
        
        # 3. 专家网络（与之前相同）
        self.experts = nn.ModuleList([
            Expert(self.emb_dim * 2, self.dropout, self.expert_size) for _ in range(self.num_experts)
        ])
        
        # 4. 多任务匹配他
        self.multi_task_tower = nn.ModuleList([
            TaskTower(self.expert_size, self.dropout) for _ in range(self.num_tasks)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        emb_paper: torch.Tensor, 
        emb_reviewer: torch.Tensor,
        task_idx: int
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        前向传播
        Args:
            emb_paper: [batch, emb_dim]
            emb_reviewer: [batch, emb_dim]
            is_real_reviewer: [batch] 辅助任务标签
        """
        # 1. 构建并编码交互特征
        interaction = torch.cat([emb_paper, emb_reviewer], dim=-1)
        interaction_feat = self.interaction_encoder(interaction)
        
        # 2. 门控逻辑：Top-K稀疏路由（核心创新）
        gates = self.multi_gating_network[task_idx](interaction_feat)  # [batch, num_experts]
        
        # 训练时添加噪声促进探索
        if self.training and self.router_noise > 0:
            noise = torch.randn_like(gates) * self.router_noise
            gates = gates + noise

        if self.topk > 0 and self.topk < self.num_experts:
            # Top-K选择：只激活k个专家
            top_k_scores, top_k_indices = torch.topk(gates, self.topk, dim=-1)

            # 稀疏门控权重（未选中的专家权重=0）
            taskgates = torch.zeros_like(gates)
            taskgates.scatter_(-1, top_k_indices, F.softmax(top_k_scores, dim=-1))

            gates = taskgates
        else:
            # 非Top-K路由：对所有专家门控权重进行softmax归一化
            gates = F.softmax(gates, dim=-1)
        
        # 3. 专家计算      
        expert_outputs = [expert(interaction_feat) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, expert_size]

        # 计算加权专家输出
        task_expert_output = torch.einsum('be,bed->bd', gates, expert_outputs)  # [batch, expert_size]
        # 通过任务塔得到最终输出
        task_output = self.multi_task_tower[task_idx](task_expert_output).squeeze(-1)  # [batch]

        return task_output, task_idx, gates

def confidence_predict_loss(
    outputs: torch.Tensor,
    batch: Dict,
    gates: Optional[torch.Tensor],
    entropy_loss_weight: float = 0.01,
) -> torch.Tensor:
    """内容驱动MoE损失函数"""
    # 主任务损失
    labels = batch["label"].float()
    pos_count = labels.sum()
    neg_count = labels.numel() - pos_count
    # 避免除零
    pos_weight = (neg_count + 1e-6) / (pos_count + 1e-6)

    # 正样本用更高权重
    sample_weight = torch.where(labels > 0.5, pos_weight, torch.tensor(1.0, device=labels.device))

    loss = F.binary_cross_entropy(outputs, labels, weight=sample_weight)
    # loss = F.binary_cross_entropy(
    #     outputs,
    #     batch["label"]
    # )
    
    # 路由器熵损失（防止门控退化为one-hot）
    if gates is None:
        return loss
    router_entropy = -(gates * torch.log(gates + 1e-6)).sum(dim=-1).mean()
    entropy_loss = -router_entropy# 最大化熵
    
    total_loss = loss + entropy_loss_weight * entropy_loss
    
    return total_loss
    
class AUCMarginLoss(nn.Module):
    """最大化AUC的同时保持排序间隔"""
    def __init__(self, margin: float = 0.5, lambda_auc: float = 0.7):
        super().__init__()
        self.margin = margin
        self.lambda_auc = lambda_auc
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        # 1. Margin Loss（保证排序正确性）
        pos_scores_expanded = pos_scores.unsqueeze(1)  # [P, 1]
        neg_scores_expanded = neg_scores.unsqueeze(0)  # [1, N]
        pair_matrix = torch.clamp(self.margin - (pos_scores_expanded - neg_scores_expanded), min=0)
        margin_loss = pair_matrix.mean()
        
        # 2. AUC Loss（最大化正样本整体排序）
        # 计算正样本对负样本的平均胜率
        pos_expanded = pos_scores.unsqueeze(1)  # [P, 1]
        neg_expanded = neg_scores.unsqueeze(0)  # [1, N]
        auc_loss = -torch.log(torch.sigmoid(pos_expanded - neg_expanded) + 1e-8).mean()
        
        return (1 - self.lambda_auc) * margin_loss + self.lambda_auc * auc_loss

# 推荐配置
criterion = AUCMarginLoss(margin=0.5, lambda_auc=0.7)  # 侧重AUC优化
    
def recommendation_ranking_loss(
    outputs: torch.Tensor,
    batch: Dict,
    margin: float = 0.5,
    lambda_auc: float = 0.7,
) -> torch.Tensor:
    """推荐排序损失函数"""
    # 主任务损失
    loss_fn = AUCMarginLoss(margin=margin, lambda_auc=lambda_auc)
    loss = loss_fn(
        outputs,
        batch["label"]
    )
    
    return loss