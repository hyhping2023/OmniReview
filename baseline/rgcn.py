"""RGCN reviewer matching baseline.

This file summarizes and extracts the RGCN-related baseline from the original
model.py and dataset/evaluate flow.

The graph is a DGL heterograph with two node types:

- author
- paper

and two reciprocal edge types:

- author --writes--> paper
- paper --written_by--> author

The evaluation flow in evaluate.py builds the graph, loads precomputed author
and paper node features, runs neighbor-sampled RGCN inference for a target paper
and candidate authors, then scores each paper/author pair with ScoringMLP.
The same scores can be evaluated with true-vs-similar, true-vs-wrong, and
similar-vs-wrong ranking tasks.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

try:
    import dgl
    import dgl.nn.pytorch as dglnn
except ImportError:  # Keep the module importable in environments without DGL.
    dgl = None
    dglnn = None


class RGCN(nn.Module):
    """Three-layer heterogeneous GCN over author-paper writing edges."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super().__init__()
        if dglnn is None:
            raise ImportError("dgl is required to instantiate RGCN.")

        self.conv1 = dglnn.HeteroGraphConv(
            {
                etype: dglnn.GraphConv(in_dim, h_dim, norm="both")
                for etype in ["writes", "written_by"]
            }
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                etype: dglnn.GraphConv(h_dim, h_dim, norm="both")
                for etype in ["writes", "written_by"]
            }
        )
        self.conv3 = dglnn.HeteroGraphConv(
            {
                etype: dglnn.GraphConv(h_dim, out_dim, norm="both")
                for etype in ["writes", "written_by"]
            }
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, blocks, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.conv1(blocks[0], x)
        h = {node_type: self.dropout(torch.relu(value)) for node_type, value in h.items()}

        h = self.conv2(blocks[1], h)
        h = {node_type: self.dropout(torch.relu(value)) for node_type, value in h.items()}

        return self.conv3(blocks[2], h)

    def inference(self, blocks, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.conv1(blocks[0], x)
        h = {node_type: torch.relu(value) for node_type, value in h.items()}

        h = self.conv2(blocks[1], h)
        h = {node_type: torch.relu(value) for node_type, value in h.items()}

        return self.conv3(blocks[2], h)


def graph_construct(writes_path: str = "data/rgcn/qwen3_rgcn_writes.npy"):
    """Construct an author-paper heterograph from integer write edges."""
    if dgl is None:
        raise ImportError("dgl is required to construct the RGCN graph.")
    if not os.path.exists(writes_path):
        raise FileNotFoundError(f"Missing graph edge file: {writes_path}")

    writes_edges = np.load(writes_path).astype(int)
    writes_authors = writes_edges[:, 0]
    writes_papers = writes_edges[:, 1]
    edges = {
        ("author", "writes", "paper"): (writes_authors, writes_papers),
        ("paper", "written_by", "author"): (writes_papers, writes_authors),
    }
    return dgl.heterograph(edges)


def graph_initialize(
    graph,
    author_feat_path: str = "data/rgcn/qwen3_rgcn_author_feats.pt",
    paper_feat_path: str = "data/rgcn/qwen3_rgcn_paper_feats.npy",
):
    """Attach precomputed node features to the author and paper nodes."""
    author_feats = torch.load(author_feat_path)
    paper_feats = torch.from_numpy(np.load(paper_feat_path))

    if author_feats.shape[0] != graph.num_nodes("author"):
        raise ValueError(
            "Author feature count does not match graph author nodes: "
            f"{author_feats.shape[0]} vs {graph.num_nodes('author')}."
        )
    if paper_feats.shape[0] != graph.num_nodes("paper"):
        raise ValueError(
            "Paper feature count does not match graph paper nodes: "
            f"{paper_feats.shape[0]} vs {graph.num_nodes('paper')}."
        )

    graph.nodes["author"].data["feat"] = author_feats
    graph.nodes["paper"].data["feat"] = paper_feats
    return graph


class ScoringMLP(nn.Module):
    """Score a target paper representation against candidate author representations."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, p_feat: torch.Tensor, a_feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(torch.cat([p_feat, a_feat], dim=-1)).squeeze(-1))


def score_author_batch(
    scorer: ScoringMLP,
    paper_feat: torch.Tensor,
    author_feats: torch.Tensor,
) -> torch.Tensor:
    """Repeat one target paper feature and score it against a batch of authors."""
    return scorer(paper_feat.repeat(author_feats.size(0), 1), author_feats)

