from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class InteractionFeatureBuilder(nn.Module):
    """Build pairwise paper-reviewer features for reranking."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.paper_norm = nn.LayerNorm(emb_dim)
        self.reviewer_norm = nn.LayerNorm(emb_dim)
        self.output_dim = emb_dim * 4 + 1

    def forward(
        self,
        emb_paper: torch.Tensor,
        emb_reviewer: torch.Tensor,
    ) -> torch.Tensor:
        if emb_paper.shape != emb_reviewer.shape:
            raise ValueError(
                "emb_paper and emb_reviewer must have the same shape, "
                f"got {tuple(emb_paper.shape)} and {tuple(emb_reviewer.shape)}."
            )
        if emb_paper.dim() != 2 or emb_paper.size(-1) != self.emb_dim:
            raise ValueError(
                f"expected [N, {self.emb_dim}] embeddings, got {tuple(emb_paper.shape)}."
            )

        paper = self.paper_norm(emb_paper)
        reviewer = self.reviewer_norm(emb_reviewer)
        abs_diff = torch.abs(paper - reviewer)
        product = paper * reviewer
        cosine = F.cosine_similarity(paper, reviewer, dim=-1, eps=1e-8).unsqueeze(-1)

        return torch.cat([paper, reviewer, abs_diff, product, cosine], dim=-1)


class InteractionMLPReranker(nn.Module):
    """Strong MLP reranker baseline over explicit interaction features."""

    def __init__(
        self,
        emb_dim: int = 1024,
        hidden_dims: Sequence[int] = (1024, 512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        self.emb_dim = emb_dim
        self.feature_builder = InteractionFeatureBuilder(emb_dim)
        self.encoder, encoder_dim = _make_mlp(
            self.feature_builder.output_dim,
            hidden_dims,
            dropout,
        )
        self.scorer = nn.Linear(encoder_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_paper: torch.Tensor,
        emb_reviewer: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        interaction_features = self.feature_builder(emb_paper, emb_reviewer)
        features = self.encoder(interaction_features)
        logits = self.scorer(features).squeeze(-1)

        return {
            "logits": logits,
            "features": features,
        }


class _Expert(nn.Module):
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


class _CGCGate(nn.Module):
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
        weights = F.softmax(logits, dim=-1)
        return weights, logits


class PLEReranker(nn.Module):
    """Single-layer PLE/CGC reranker with shared and task-specific experts."""

    def __init__(
        self,
        emb_dim: int = 1024,
        num_tasks: int = 2,
        num_shared_experts: int = 2,
        num_task_experts: int = 2,
        expert_dim: int = 256,
        expert_hidden_dims: Sequence[int] = (512,),
        tower_hidden_dims: Sequence[int] = (256,),
        dropout: float = 0.1,
        router_noise: float = 0.0,
    ):
        super().__init__()
        if num_tasks < 1:
            raise ValueError("num_tasks must be >= 1.")
        if num_shared_experts < 1:
            raise ValueError("num_shared_experts must be >= 1.")
        if num_task_experts < 1:
            raise ValueError("num_task_experts must be >= 1.")

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.expert_dim = expert_dim
        self.router_noise = router_noise

        self.feature_builder = InteractionFeatureBuilder(emb_dim)
        feature_dim = self.feature_builder.output_dim

        self.shared_experts = nn.ModuleList(
            [
                _Expert(feature_dim, expert_dim, expert_hidden_dims, dropout)
                for _ in range(num_shared_experts)
            ]
        )
        self.task_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _Expert(feature_dim, expert_dim, expert_hidden_dims, dropout)
                        for _ in range(num_task_experts)
                    ]
                )
                for _ in range(num_tasks)
            ]
        )

        selected_experts = num_task_experts + num_shared_experts
        self.gates = nn.ModuleList(
            [
                _CGCGate(feature_dim, selected_experts, dropout, router_noise=router_noise)
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

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_paper: torch.Tensor,
        emb_reviewer: torch.Tensor,
        task_idx: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if task_idx < 0 or task_idx >= self.num_tasks:
            raise ValueError(f"task_idx must be in [0, {self.num_tasks}), got {task_idx}.")

        interaction_features = self.feature_builder(emb_paper, emb_reviewer)

        selected_outputs = [
            expert(interaction_features) for expert in self.task_experts[task_idx]
        ]
        selected_outputs.extend(
            expert(interaction_features) for expert in self.shared_experts
        )
        expert_outputs = torch.stack(selected_outputs, dim=1)

        gate_weights, gate_logits = self.gates[task_idx](interaction_features)
        features = torch.einsum("be,bed->bd", gate_weights, expert_outputs)
        tower_features = self.towers[task_idx](features)
        logits = self.scorers[task_idx](tower_features).squeeze(-1)

        return {
            "logits": logits,
            "features": features,
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
        }


class QueryPairwiseRankLoss(nn.Module):
    """Query-level pairwise logistic ranking loss over relevance labels."""

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        group_sizes: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        scores = scores.reshape(-1)
        labels = labels.reshape(-1).to(scores.device)

        if isinstance(group_sizes, torch.Tensor):
            group_sizes = [int(size) for size in group_sizes.detach().cpu().tolist()]
        else:
            group_sizes = [int(size) for size in group_sizes]

        if sum(group_sizes) != scores.numel():
            raise ValueError(
                f"sum(group_sizes) must equal scores.numel(), got "
                f"{sum(group_sizes)} and {scores.numel()}."
            )
        if labels.numel() != scores.numel():
            raise ValueError(
                f"labels.numel() must equal scores.numel(), got "
                f"{labels.numel()} and {scores.numel()}."
            )

        losses: list[torch.Tensor] = []
        offset = 0
        for group_size in group_sizes:
            if group_size <= 0:
                raise ValueError(f"group sizes must be positive, got {group_size}.")

            group_scores = scores[offset : offset + group_size]
            group_labels = labels[offset : offset + group_size]
            offset += group_size

            pair_mask = group_labels.unsqueeze(1) > group_labels.unsqueeze(0)
            if not pair_mask.any():
                continue

            score_diff = group_scores.unsqueeze(1) - group_scores.unsqueeze(0)
            losses.append(F.softplus(-score_diff[pair_mask]).mean())

        if not losses:
            return scores.sum() * 0.0

        return torch.stack(losses).mean()


class BinaryLogitLoss(nn.Module):
    """Binary BCE loss for logits, useful for coarse true-vs-nontrue training."""

    def __init__(self, pos_weight: float | torch.Tensor | None = None):
        super().__init__()
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.as_tensor(pos_weight, dtype=torch.float32),
            )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)

        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            labels.reshape(-1).to(device=logits.device, dtype=logits.dtype),
            weight=weight,
            pos_weight=pos_weight,
        )


def _self_check() -> None:
    torch.manual_seed(0)
    emb_dim = 16
    batch_size = 8

    emb_paper = torch.randn(batch_size, emb_dim)
    emb_reviewer = torch.randn(batch_size, emb_dim)

    mlp_model = InteractionMLPReranker(
        emb_dim=emb_dim,
        hidden_dims=(32, 16),
        dropout=0.1,
    )
    mlp_outputs = mlp_model(emb_paper, emb_reviewer)
    assert mlp_outputs["logits"].shape == (batch_size,)
    assert mlp_outputs["features"].shape == (batch_size, 16)

    ple_model = PLEReranker(
        emb_dim=emb_dim,
        num_tasks=2,
        num_shared_experts=2,
        num_task_experts=2,
        expert_dim=12,
        expert_hidden_dims=(24,),
        tower_hidden_dims=(12,),
        dropout=0.1,
    )
    for task_idx in (0, 1):
        ple_outputs = ple_model(emb_paper, emb_reviewer, task_idx=task_idx)
        assert ple_outputs["logits"].shape == (batch_size,)
        assert ple_outputs["features"].shape == (batch_size, 12)
        assert ple_outputs["gate_weights"].shape == (batch_size, 4)

    scores = torch.randn(6, requires_grad=True)
    labels = torch.tensor([2, 1, 0, 2, 1, 0], dtype=torch.float32)
    group_sizes = torch.tensor([3, 3])
    rank_loss = QueryPairwiseRankLoss()(scores, labels, group_sizes)
    assert torch.isfinite(rank_loss)
    rank_loss.backward()
    assert scores.grad is not None

    binary_loss = BinaryLogitLoss()(
        torch.randn(4),
        torch.tensor([1, 0, 1, 0], dtype=torch.float32),
    )
    assert torch.isfinite(binary_loss)

    print("rerankers.py self-check passed")


if __name__ == "__main__":
    _self_check()
