from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .reviewer_paper_encoders import ReviewerPaperMeanEncoder, ReviewerPaperTransformerEncoder
    from .rerankers import InteractionFeatureBuilder
except ImportError:
    from reviewer_paper_encoders import ReviewerPaperMeanEncoder, ReviewerPaperTransformerEncoder
    from rerankers import InteractionFeatureBuilder


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> tuple[nn.Module, int]:
    layers: list[nn.Module] = []
    current_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim

    if not layers:
        return nn.Identity(), current_dim
    return nn.Sequential(*layers), current_dim


class _MMoEExpert(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        mlp, mlp_dim = _make_mlp(input_dim, hidden_dims, dropout)
        self.network = nn.Sequential(
            mlp,
            nn.Linear(mlp_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _MMoEGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        dropout: float,
        hidden_dim: int = 128,
        router_noise: float = 0.0,
    ):
        super().__init__()
        self.router_noise = router_noise
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(x)
        if self.training and self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        dense_weights = F.softmax(logits, dim=-1)
        return dense_weights, logits


class DualPathPaperSetMMoE(nn.Module):
    """Dual-path paper-set MMoE with paper-set encoders on both qwen and wollm paths."""

    def __init__(
        self,
        emb_dim: int = 1024,
        qwen_model_dim: int = 256,
        max_papers: int = 10,
        reviewer_encoder_type: str = "mean",
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        fusion_hidden_dims: Sequence[int] = (512, 256),
        num_tasks: int = 3,
        num_experts: int = 4,
        expert_dim: int = 128,
        expert_hidden_dims: Sequence[int] = (512,),
        tower_hidden_dims: Sequence[int] = (256,),
        dropout: float = 0.2,
        router_noise: float = 0.05,
        topk: int | None = None,
    ):
        super().__init__()
        if num_tasks < 1:
            raise ValueError("num_tasks must be >= 1.")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1.")
        if topk is not None and topk < 1:
            raise ValueError("topk must be >= 1 when provided.")

        self.emb_dim = emb_dim
        self.qwen_model_dim = qwen_model_dim
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.topk = num_experts if topk is None else min(topk, num_experts)

        if reviewer_encoder_type not in {"transformer", "mean"}:
            raise ValueError(
                "reviewer_encoder_type must be one of {'transformer', 'mean'}, "
                f"got {reviewer_encoder_type}."
            )

        self.qwen_encoder = self._build_reviewer_encoder(
            reviewer_encoder_type=reviewer_encoder_type,
            emb_dim=emb_dim,
            model_dim=qwen_model_dim,
            max_papers=max_papers,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        self.wollm_encoder = self._build_reviewer_encoder(
            reviewer_encoder_type=reviewer_encoder_type,
            emb_dim=emb_dim,
            model_dim=qwen_model_dim,
            max_papers=max_papers,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

        self.qwen_feature_builder = InteractionFeatureBuilder(qwen_model_dim)
        self.wollm_feature_builder = InteractionFeatureBuilder(qwen_model_dim)
        fusion_input_dim = (
            self.qwen_feature_builder.output_dim
            + self.wollm_feature_builder.output_dim
        )
        self.fusion_encoder, fusion_dim = _make_mlp(
            fusion_input_dim,
            fusion_hidden_dims,
            dropout,
        )

        self.experts = nn.ModuleList(
            [
                _MMoEExpert(fusion_dim, expert_dim, expert_hidden_dims, dropout)
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [
                _MMoEGate(
                    input_dim=fusion_dim,
                    num_experts=num_experts,
                    dropout=dropout,
                    router_noise=router_noise,
                )
                for _ in range(num_tasks)
            ]
        )
        self.towers = nn.ModuleList()
        self.scorers = nn.ModuleList()
        for _ in range(num_tasks):
            tower, tower_dim = _make_mlp(expert_dim, tower_hidden_dims, dropout)
            self.towers.append(tower)
            self.scorers.append(nn.Linear(tower_dim, 1))

        self._init_weights()

    @staticmethod
    def _build_reviewer_encoder(
        reviewer_encoder_type: str,
        emb_dim: int,
        model_dim: int,
        max_papers: int,
        num_transformer_layers: int,
        num_attention_heads: int,
        dropout: float,
    ) -> nn.Module:
        if reviewer_encoder_type == "transformer":
            return ReviewerPaperTransformerEncoder(
                emb_dim=emb_dim,
                model_dim=model_dim,
                max_papers=max_papers,
                num_layers=num_transformer_layers,
                num_heads=num_attention_heads,
                dropout=dropout,
            )
        return ReviewerPaperMeanEncoder(
            emb_dim=emb_dim,
            model_dim=model_dim,
            max_papers=max_papers,
            dropout=dropout,
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def build_fused_features(
        self,
        emb_paper: torch.Tensor,
        reviewer_paper_embs: torch.Tensor,
        reviewer_paper_mask: torch.Tensor,
        emb_wollm_paper: torch.Tensor,
        wollm_reviewer_paper_embs: torch.Tensor,
        wollm_reviewer_paper_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qwen_outputs = self.qwen_encoder(
            emb_paper=emb_paper,
            reviewer_paper_embs=reviewer_paper_embs,
            reviewer_paper_mask=reviewer_paper_mask,
        )
        wollm_outputs = self.wollm_encoder(
            emb_paper=emb_wollm_paper,
            reviewer_paper_embs=wollm_reviewer_paper_embs,
            reviewer_paper_mask=wollm_reviewer_paper_mask,
        )
        qwen_features = self.qwen_feature_builder(
            qwen_outputs["paper_repr"],
            qwen_outputs["reviewer_repr"],
        )
        wollm_features = self.wollm_feature_builder(
            wollm_outputs["paper_repr"],
            wollm_outputs["reviewer_repr"],
        )
        # Fuse the two embedding spaces after each path has built the same interaction
        # features, so downstream experts see one shared representation.
        fused_features = self.fusion_encoder(
            torch.cat([qwen_features, wollm_features], dim=-1)
        )
        return (
            fused_features,
            qwen_outputs["paper_attention_weights"],
            wollm_outputs["paper_attention_weights"],
        )

    def _route(
        self,
        task_idx: int,
        fused_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dense_weights, gate_logits = self.gates[task_idx](fused_features)

        # topk >= num_experts keeps dense MMoE routing; smaller topk sparsifies the gate
        # and renormalizes only the selected experts.
        if self.topk >= self.num_experts:
            gate_indices = torch.arange(
                self.num_experts,
                device=fused_features.device,
            ).unsqueeze(0).expand(fused_features.size(0), -1)
            selected_weights = dense_weights
            full_weights = dense_weights
            return selected_weights, gate_indices, full_weights, gate_logits

        topk_weights, gate_indices = torch.topk(dense_weights, self.topk, dim=-1)
        selected_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        full_weights = torch.zeros_like(dense_weights)
        full_weights.scatter_(dim=-1, index=gate_indices, src=selected_weights)
        return selected_weights, gate_indices, full_weights, gate_logits

    def forward(
        self,
        emb_paper: torch.Tensor,
        reviewer_paper_embs: torch.Tensor,
        reviewer_paper_mask: torch.Tensor,
        emb_wollm_paper: torch.Tensor,
        wollm_reviewer_paper_embs: torch.Tensor,
        wollm_reviewer_paper_mask: torch.Tensor,
        task_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        if task_idx < 0 or task_idx >= self.num_tasks:
            raise ValueError(f"task_idx must be in [0, {self.num_tasks}), got {task_idx}.")

        fused_features, attention_weights, wollm_attention_weights = self.build_fused_features(
            emb_paper=emb_paper,
            reviewer_paper_embs=reviewer_paper_embs,
            reviewer_paper_mask=reviewer_paper_mask,
            emb_wollm_paper=emb_wollm_paper,
            wollm_reviewer_paper_embs=wollm_reviewer_paper_embs,
            wollm_reviewer_paper_mask=wollm_reviewer_paper_mask,
        )

        expert_outputs = torch.stack([expert(fused_features) for expert in self.experts], dim=1)
        gate_weights, gate_indices, gate_full_weights, gate_logits = self._route(
            task_idx=task_idx,
            fused_features=fused_features,
        )
        features = torch.einsum("be,bed->bd", gate_full_weights, expert_outputs)
        tower_features = self.towers[task_idx](features)
        logits = self.scorers[task_idx](tower_features).squeeze(-1)

        return {
            "logits": logits,
            "features": features,
            "fused_features": fused_features,
            "gate_weights": gate_weights,
            "gate_indices": gate_indices,
            "gate_full_weights": gate_full_weights,
            "gate_dense_weights": F.softmax(gate_logits, dim=-1),
            "gate_logits": gate_logits,
            "paper_attention_weights": attention_weights,
            "wollm_attention_weights": wollm_attention_weights,
        }


def _self_check() -> None:
    torch.manual_seed(0)

    model = DualPathPaperSetMMoE(
        emb_dim=16,
        qwen_model_dim=8,
        max_papers=5,
        reviewer_encoder_type="mean",
        fusion_hidden_dims=(12,),
        num_tasks=3,
        num_experts=4,
        expert_dim=6,
        expert_hidden_dims=(10,),
        tower_hidden_dims=(6,),
        dropout=0.1,
        router_noise=0.01,
        topk=100,
    )

    outputs = model(
        emb_paper=torch.randn(4, 16),
        reviewer_paper_embs=torch.randn(4, 5, 16),
        reviewer_paper_mask=torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        emb_wollm_paper=torch.randn(4, 16),
        wollm_reviewer_paper_embs=torch.randn(4, 5, 16),
        wollm_reviewer_paper_mask=torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        task_idx=1,
    )

    assert outputs["logits"].shape == (4,)
    assert outputs["features"].shape == (4, 6)
    assert outputs["fused_features"].shape == (4, 12)
    assert outputs["gate_weights"].shape == (4, 4)
    assert outputs["gate_indices"].shape == (4, 4)
    assert outputs["gate_full_weights"].shape == (4, 4)
    assert outputs["gate_dense_weights"].shape == (4, 4)
    assert outputs["paper_attention_weights"].shape == (4, 5)
    assert outputs["wollm_attention_weights"].shape == (4, 5)
    assert torch.isfinite(outputs["logits"]).all()

    print("paper_set_dual_path_mmoe.py self-check passed")


if __name__ == "__main__":
    _self_check()
