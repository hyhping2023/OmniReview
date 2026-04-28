from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .rerankers import InteractionFeatureBuilder
except ImportError:
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


class ReviewerPaperTransformerEncoder(nn.Module):
    """Encode reviewer papers conditioned on the target paper."""

    def __init__(
        self,
        emb_dim: int = 1024,
        model_dim: int = 256,
        max_papers: int = 10,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        if max_papers < 1:
            raise ValueError("max_papers must be >= 1.")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim must be divisible by num_heads, got {model_dim} and {num_heads}."
            )

        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.max_papers = max_papers

        self.paper_projection = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reviewer_projection = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_embedding = nn.Parameter(torch.zeros(max_papers, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)

        self.query_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.key_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.value_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_paper: torch.Tensor,
        reviewer_paper_embs: torch.Tensor,
        reviewer_paper_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if emb_paper.dim() != 2 or emb_paper.size(-1) != self.emb_dim:
            raise ValueError(
                f"emb_paper must have shape [N, {self.emb_dim}], got {tuple(emb_paper.shape)}."
            )
        if reviewer_paper_embs.dim() != 3:
            raise ValueError(
                "reviewer_paper_embs must have shape [N, max_papers, emb_dim], "
                f"got {tuple(reviewer_paper_embs.shape)}."
            )
        if reviewer_paper_embs.size(0) != emb_paper.size(0):
            raise ValueError("emb_paper and reviewer_paper_embs must share batch size.")
        if reviewer_paper_embs.size(1) != self.max_papers:
            raise ValueError(
                f"expected max_papers={self.max_papers}, got {reviewer_paper_embs.size(1)}."
            )
        if reviewer_paper_embs.size(2) != self.emb_dim:
            raise ValueError(
                f"expected reviewer paper emb_dim={self.emb_dim}, got {reviewer_paper_embs.size(2)}."
            )
        if reviewer_paper_mask.shape != reviewer_paper_embs.shape[:2]:
            raise ValueError(
                "reviewer_paper_mask must have shape [N, max_papers], "
                f"got {tuple(reviewer_paper_mask.shape)}."
            )

        mask = reviewer_paper_mask.to(device=emb_paper.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Each reviewer must have at least one valid paper embedding.")

        paper_repr = self.paper_projection(emb_paper)
        reviewer_tokens = self.reviewer_projection(reviewer_paper_embs)
        reviewer_tokens = reviewer_tokens + self.position_embedding.unsqueeze(0)

        encoded_tokens = self.transformer(
            reviewer_tokens,
            src_key_padding_mask=~mask,
        )
        encoded_tokens = self.output_norm(encoded_tokens)

        query = self.query_projection(paper_repr).unsqueeze(1)
        keys = self.key_projection(encoded_tokens)
        values = self.value_projection(encoded_tokens)

        attention_logits = (query * keys).sum(dim=-1) / (self.model_dim ** 0.5)
        attention_logits = attention_logits.masked_fill(
            ~mask,
            torch.finfo(attention_logits.dtype).min,
        )
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = attention_weights.masked_fill(~mask, 0.0)
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        pooled_reviewer = torch.bmm(
            self.attention_dropout(attention_weights).unsqueeze(1),
            values,
        ).squeeze(1)
        reviewer_repr = self.output_norm(pooled_reviewer)

        return {
            "paper_repr": paper_repr,
            "reviewer_repr": reviewer_repr,
            "paper_attention_weights": attention_weights,
        }


class ReviewerPaperMeanEncoder(nn.Module):
    """Ablation encoder using masked mean pooling over reviewer papers."""

    def __init__(
        self,
        emb_dim: int = 1024,
        model_dim: int = 256,
        max_papers: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        if max_papers < 1:
            raise ValueError("max_papers must be >= 1.")

        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.max_papers = max_papers

        self.paper_projection = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reviewer_projection = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_paper: torch.Tensor,
        reviewer_paper_embs: torch.Tensor,
        reviewer_paper_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if emb_paper.dim() != 2 or emb_paper.size(-1) != self.emb_dim:
            raise ValueError(
                f"emb_paper must have shape [N, {self.emb_dim}], got {tuple(emb_paper.shape)}."
            )
        if reviewer_paper_embs.dim() != 3:
            raise ValueError(
                "reviewer_paper_embs must have shape [N, max_papers, emb_dim], "
                f"got {tuple(reviewer_paper_embs.shape)}."
            )
        if reviewer_paper_embs.size(0) != emb_paper.size(0):
            raise ValueError("emb_paper and reviewer_paper_embs must share batch size.")
        if reviewer_paper_embs.size(1) != self.max_papers:
            raise ValueError(
                f"expected max_papers={self.max_papers}, got {reviewer_paper_embs.size(1)}."
            )
        if reviewer_paper_embs.size(2) != self.emb_dim:
            raise ValueError(
                f"expected reviewer paper emb_dim={self.emb_dim}, got {reviewer_paper_embs.size(2)}."
            )
        if reviewer_paper_mask.shape != reviewer_paper_embs.shape[:2]:
            raise ValueError(
                "reviewer_paper_mask must have shape [N, max_papers], "
                f"got {tuple(reviewer_paper_mask.shape)}."
            )

        mask = reviewer_paper_mask.to(device=emb_paper.device, dtype=emb_paper.dtype)
        valid_counts = mask.sum(dim=1, keepdim=True)
        if not (valid_counts > 0).all():
            raise ValueError("Each reviewer must have at least one valid paper embedding.")

        attention_weights = mask / valid_counts.clamp_min(1.0)
        pooled_reviewer = torch.bmm(
            attention_weights.unsqueeze(1),
            reviewer_paper_embs,
        ).squeeze(1)

        paper_repr = self.paper_projection(emb_paper)
        reviewer_repr = self.reviewer_projection(pooled_reviewer)

        return {
            "paper_repr": paper_repr,
            "reviewer_repr": reviewer_repr,
            "paper_attention_weights": attention_weights,
        }


