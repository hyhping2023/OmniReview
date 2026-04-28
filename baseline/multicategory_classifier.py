"""Category-specific two-tower reviewer matching baseline.

This file summarizes and extracts the MultiCategoryClassifier-related baseline
from the original model.py and dataset.py:

- ArticleTowerV1 encodes target paper features.
- CandidateTowerV1 encodes each candidate reviewer from historical papers,
  years, and affiliation embeddings.
- RegressionMLP scores the concatenated paper/reviewer vectors.
- MultiCategoryClassifier keeps one article tower, candidate tower, and scorer
  per research category.

The expected evaluation dataset follows TowerTestDatasetV1 from dataset.py:

- paper: abstract embedding, venue embedding, keyword embedding, year, category
- reviewers: true reviewers
- similar_candidates: hard negative reviewers
- wrong_candidates: easy negative reviewers

Evaluation normally scores each group with the same target-paper input, then
builds ranking tasks:

- true reviewers vs similar candidates
- true reviewers vs wrong candidates
- similar candidates vs wrong candidates
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid_pos_encoding(max_seq: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_seq, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(max_seq, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class ArticleTowerV1(nn.Module):
    """Encode a target paper from abstract, keyword, venue, and year features."""

    def __init__(
        self,
        a_dim: int = 768,
        k_dim: int = 128,
        v_dim: int = 32,
        hidden: int = 256,
        min_year: int = 1951,
        max_year: int = 2050,
        max_seq: int = 512,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.keywords_emb = nn.Linear(384, k_dim)
        self.venue_emb = nn.Linear(384, v_dim)
        self.d_model = a_dim + k_dim + v_dim
        self.net = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.year_min = min_year
        self.year_max = max_year
        self.year_range = max_year - min_year + 1
        self.max_seq = max_seq
        self.device = device

    def forward(
        self,
        abstract_embeddings: torch.Tensor,
        keywords: torch.Tensor,
        venue: torch.Tensor,
        year: torch.Tensor,
    ) -> torch.Tensor:
        pos_idx = ((year - self.year_min) / (self.year_range - 1) * (self.max_seq - 1))
        pos_idx = pos_idx.long().clamp(0, self.max_seq - 1).to(self.device)
        year_pos_emb = sinusoid_pos_encoding(self.max_seq, self.d_model).to(self.device)[pos_idx]

        keyword_vec = self.keywords_emb(keywords)
        venue_vec = self.venue_emb(venue)
        features = torch.cat([abstract_embeddings, keyword_vec, venue_vec], dim=-1)
        return F.normalize(self.net(features + year_pos_emb), dim=-1)


class CandidateTowerV1(nn.Module):
    """Encode reviewer paper sets with a Transformer and a CLS pooling token."""

    def __init__(
        self,
        paper_dim: int = 768,
        d_aff: int = 32,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_year: int = 2050,
        min_year: int = 1951,
        max_seq: int = 512,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.device = device
        self.year_min = min_year
        self.year_max = max_year
        self.year_range = max_year - min_year + 1

        self.aff_emb = nn.Linear(384, d_aff)
        self.paper_proj = nn.Linear(paper_dim + d_aff, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pooler = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 256)

    def forward(
        self,
        paper_embs: torch.Tensor,
        years: torch.Tensor,
        aff_embs: torch.Tensor,
        paper_mask: torch.Tensor,
        year_mask: torch.Tensor,
    ) -> torch.Tensor:
        del paper_mask
        batch_size, candidate_count, paper_count, emb_dim = paper_embs.shape

        paper_embs = paper_embs.reshape(batch_size * candidate_count, paper_count, emb_dim)
        years = years.reshape(batch_size * candidate_count, paper_count)
        aff_embs = (
            aff_embs.reshape(batch_size * candidate_count, aff_embs.shape[-1])
            .unsqueeze(1)
            .expand(-1, paper_count, -1)
        )
        year_mask = year_mask.reshape(batch_size * candidate_count, paper_count)

        pos_idx = ((years - self.year_min) / (self.year_range - 1) * (self.max_seq - 1))
        pos_idx = pos_idx.long().clamp(0, self.max_seq - 1)
        year_pos_emb = sinusoid_pos_encoding(self.max_seq, self.d_model).to(self.device)[pos_idx]

        aff_vec = self.aff_emb(aff_embs)
        token_feat = self.paper_proj(torch.cat([paper_embs, aff_vec], dim=-1))
        token_feat = token_feat + year_pos_emb

        flat_count = batch_size * candidate_count
        cls_tokens = self.cls_token.expand(flat_count, -1, -1)
        tokens = torch.cat([cls_tokens, token_feat], dim=1)

        cls_mask = torch.zeros(flat_count, 1, dtype=torch.bool, device=self.device)
        key_padding_mask = torch.cat([cls_mask, ~year_mask], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        cls_vec = F.relu(self.pooler(encoded[:, 0, :]))
        reviewer_vec = self.out_proj(cls_vec)
        return F.normalize(reviewer_vec, dim=-1).reshape(batch_size, candidate_count, -1)


class RegressionMLP(nn.Module):
    """Score concatenated article and candidate vectors as a confidence value."""

    def __init__(self, input_dim: int = 256 * 2, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class MultiCategoryClassifier(nn.Module):
    """One two-tower scorer per category."""

    def __init__(
        self,
        category: list[str],
        article_params: dict,
        candidate_params: dict,
        regression_params: dict,
    ):
        super().__init__()
        self.category_classifier = nn.ModuleDict()
        for cat in category:
            self.category_classifier[cat] = nn.ModuleList(
                [
                    ArticleTowerV1(**article_params),
                    CandidateTowerV1(**candidate_params),
                    RegressionMLP(**regression_params),
                ]
            )

    def forward(
        self,
        category: str,
        article_inputs: dict[str, torch.Tensor],
        candidate_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if category not in self.category_classifier:
            raise KeyError(f"Category {category} is not in model categories.")

        article_tower, candidate_tower, regressor = self.category_classifier[category]
        article_vec = article_tower(**article_inputs)
        candidate_vecs = candidate_tower(**candidate_inputs)

        _, candidate_count, _ = candidate_vecs.shape
        article_repeated = article_vec.unsqueeze(1).expand_as(candidate_vecs)
        combined = torch.cat(
            [
                article_repeated.reshape(-1, article_repeated.shape[-1]),
                candidate_vecs.reshape(-1, candidate_vecs.shape[-1]),
            ],
            dim=-1,
        )
        return regressor(combined).reshape(-1, candidate_count)

