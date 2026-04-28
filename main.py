import argparse
import json
import os
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

import reviewer_training_utils as training_utils
from models.paper_set_dual_path_mmoe import DualPathPaperSetMMoE
from models.rerankers import BinaryLogitLoss, QueryPairwiseRankLoss


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = os.path.join(BASE_DIR, "data/checkpoints")


def resolve_qwen_cache(args: argparse.Namespace, split: str) -> str:
    shared_cache = getattr(args, "paper_embedding_cache", None)
    if shared_cache:
        return shared_cache
    return getattr(args, f"{split}_paper_cache")


def resolve_wollm_cache(args: argparse.Namespace, split: str) -> str:
    shared_cache = getattr(args, "wollm_embedding_cache", None)
    if shared_cache:
        return shared_cache
    return getattr(args, f"{split}_wollm_cache")


class DualPathPaperSetDataset(Dataset):
    """Tri-task dual-path dataset using reviewer paper sets on both qwen and wollm branches."""

    def __init__(
        self,
        df_path: str,
        author_papers_path: str,
        qwen_cache_path: str,
        wollm_cache_path: str,
        split: str,
        args: argparse.Namespace,
    ):
        self.df = pd.read_parquet(df_path).reset_index(drop=True)
        self.author_publications = training_utils.load_author_publications(author_papers_path)
        self.split = split
        self.use_full_pool = split == "test"
        self.emb_dim = args.emb_dim
        self.max_reviewer_papers = args.max_reviewer_papers
        self.citation_topk = args.citation_topk
        self.indices = training_utils.sample_dataframe_indices(
            self.df,
            args.sample_ratio if split != "test" else 1.0,
            args.seed,
        )

        # Pre-compute the exact paper ids used by this split so cache loading can stream
        # only relevant embeddings instead of materializing the full cache file.
        required_paper_ids = training_utils.collect_required_paper_ids(
            df=self.df,
            indices=self.indices,
            author_publications=self.author_publications,
            use_full_pool=self.use_full_pool,
            max_papers=self.max_reviewer_papers,
            citation_topk=self.citation_topk,
        )
        self.qwen_cache = training_utils.load_filtered_paper_cache(
            cache_path=qwen_cache_path,
            required_paper_ids=required_paper_ids,
            emb_dim=self.emb_dim,
        )
        self.wollm_cache = training_utils.load_filtered_paper_cache(
            cache_path=wollm_cache_path,
            required_paper_ids=required_paper_ids,
            emb_dim=self.emb_dim,
        )

        self.missing_author_count = 0
        self.missing_qwen_reviewer_paper_count = 0
        self.missing_wollm_reviewer_paper_count = 0
        self.empty_qwen_reviewer_count = 0
        self.empty_wollm_reviewer_count = 0

        self._selected_publication_cache: dict[str, list[dict] | None] = {}
        self._qwen_reviewer_cache: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {}
        self._wollm_reviewer_cache: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {}

        print(
            f"{split} dual-path paper-set dataset loaded: rows={len(self.df)}, "
            f"active_rows={len(self.indices)}, full_pool={self.use_full_pool}, "
            f"authors={len(self.author_publications)}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def summary(self) -> str:
        return (
            f"missing_author={self.missing_author_count}, "
            f"missing_qwen_reviewer_paper={self.missing_qwen_reviewer_paper_count}, "
            f"missing_wollm_reviewer_paper={self.missing_wollm_reviewer_paper_count}, "
            f"empty_qwen_reviewer={self.empty_qwen_reviewer_count}, "
            f"empty_wollm_reviewer={self.empty_wollm_reviewer_count}"
        )

    def _paper_embedding(
        self,
        cache: dict[str, np.ndarray],
        paper_id: str,
        cache_name: str,
    ) -> np.ndarray:
        key = training_utils.paper_cache_key(paper_id)
        if key not in cache:
            raise ValueError(f"Target {cache_name} paper embedding not found in cache: {key}")
        return cache[key]

    def _selected_publications(self, reviewer_id: str) -> list[dict] | None:
        reviewer_id = str(reviewer_id)
        if reviewer_id in self._selected_publication_cache:
            return self._selected_publication_cache[reviewer_id]

        publications = self.author_publications.get(reviewer_id)
        if not publications:
            self.missing_author_count += 1
            self._selected_publication_cache[reviewer_id] = None
            return None

        selected = training_utils.select_candidate_papers(
            publications,
            topk=self.max_reviewer_papers,
            citation_topk=self.citation_topk,
        )
        self._selected_publication_cache[reviewer_id] = selected
        return selected

    def _reviewer_paper_tensor(
        self,
        reviewer_id: str,
        cache: dict[str, np.ndarray],
        reviewer_cache: dict[str, tuple[torch.Tensor, torch.Tensor] | None],
        missing_attr: str,
        empty_attr: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        reviewer_id = str(reviewer_id)
        if reviewer_id in reviewer_cache:
            return reviewer_cache[reviewer_id]

        # Build a fixed-size paper matrix plus mask for one reviewer. Missing reviewer
        # papers are skipped, but an entirely empty reviewer is removed from the group.
        selected = self._selected_publications(reviewer_id)
        if not selected:
            reviewer_cache[reviewer_id] = None
            return None

        embeddings: list[np.ndarray] = []
        for paper in selected:
            key = training_utils.paper_cache_key(str(paper.get("id", "")))
            if key in cache:
                embeddings.append(cache[key])
            else:
                setattr(self, missing_attr, getattr(self, missing_attr) + 1)

        if not embeddings:
            setattr(self, empty_attr, getattr(self, empty_attr) + 1)
            reviewer_cache[reviewer_id] = None
            return None

        matrix = np.zeros((self.max_reviewer_papers, self.emb_dim), dtype=np.float32)
        mask = np.zeros((self.max_reviewer_papers,), dtype=np.float32)
        count = min(len(embeddings), self.max_reviewer_papers)
        matrix[:count] = np.asarray(embeddings[:count], dtype=np.float32)
        mask[:count] = 1.0
        built = (
            torch.tensor(matrix, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )
        reviewer_cache[reviewer_id] = built
        return built

    def _reviewer_qwen_papers(self, reviewer_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._reviewer_paper_tensor(
            reviewer_id=reviewer_id,
            cache=self.qwen_cache,
            reviewer_cache=self._qwen_reviewer_cache,
            missing_attr="missing_qwen_reviewer_paper_count",
            empty_attr="empty_qwen_reviewer_count",
        )

    def _reviewer_wollm_papers(self, reviewer_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._reviewer_paper_tensor(
            reviewer_id=reviewer_id,
            cache=self.wollm_cache,
            reviewer_cache=self._wollm_reviewer_cache,
            missing_attr="missing_wollm_reviewer_paper_count",
            empty_attr="empty_wollm_reviewer_count",
        )

    def _candidate_group(
        self,
        reviewer_ids: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qwen_paper_sets = []
        qwen_masks = []
        wollm_paper_sets = []
        wollm_masks = []
        for reviewer_id in reviewer_ids:
            qwen_built = self._reviewer_qwen_papers(str(reviewer_id))
            wollm_built = self._reviewer_wollm_papers(str(reviewer_id))
            if qwen_built is None or wollm_built is None:
                continue
            qwen_papers, qwen_mask = qwen_built
            wollm_papers, wollm_mask = wollm_built
            qwen_paper_sets.append(qwen_papers)
            qwen_masks.append(qwen_mask)
            wollm_paper_sets.append(wollm_papers)
            wollm_masks.append(wollm_mask)

        if not qwen_paper_sets:
            return (
                torch.empty((0, self.max_reviewer_papers, self.emb_dim), dtype=torch.float32),
                torch.empty((0, self.max_reviewer_papers), dtype=torch.float32),
                torch.empty((0, self.max_reviewer_papers, self.emb_dim), dtype=torch.float32),
                torch.empty((0, self.max_reviewer_papers), dtype=torch.float32),
            )

        return (
            torch.stack(qwen_paper_sets, dim=0),
            torch.stack(qwen_masks, dim=0),
            torch.stack(wollm_paper_sets, dim=0),
            torch.stack(wollm_masks, dim=0),
        )

    def _task_tensors(
        self,
        positive_ids: Sequence[str],
        negative_ids: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        # Each task is represented as one candidate group: positives first, negatives
        # second, with labels aligned to the concatenated reviewer tensors.
        (
            positive_qwen_sets,
            positive_qwen_masks,
            positive_wollm_sets,
            positive_wollm_masks,
        ) = self._candidate_group(positive_ids)
        (
            negative_qwen_sets,
            negative_qwen_masks,
            negative_wollm_sets,
            negative_wollm_masks,
        ) = self._candidate_group(negative_ids)

        if positive_qwen_sets.size(0) == 0 or negative_qwen_sets.size(0) == 0:
            return {
                "qwen_reviewers": torch.empty((0, self.max_reviewer_papers, self.emb_dim), dtype=torch.float32),
                "qwen_masks": torch.empty((0, self.max_reviewer_papers), dtype=torch.float32),
                "wollm_reviewers": torch.empty((0, self.max_reviewer_papers, self.emb_dim), dtype=torch.float32),
                "wollm_masks": torch.empty((0, self.max_reviewer_papers), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.float32),
            }

        return {
            "qwen_reviewers": torch.cat([positive_qwen_sets, negative_qwen_sets], dim=0),
            "qwen_masks": torch.cat([positive_qwen_masks, negative_qwen_masks], dim=0),
            "wollm_reviewers": torch.cat([positive_wollm_sets, negative_wollm_sets], dim=0),
            "wollm_masks": torch.cat([positive_wollm_masks, negative_wollm_masks], dim=0),
            "labels": torch.tensor(
                [1.0] * positive_qwen_sets.size(0) + [0.0] * negative_qwen_sets.size(0),
                dtype=torch.float32,
            ),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source_idx = self.indices[idx]
        row = self.df.iloc[source_idx]
        reviewer_ids, wrong_ids, similar_ids = training_utils.row_candidate_ids(
            row,
            source_idx=source_idx,
            use_full_pool=self.use_full_pool,
        )

        qwen_paper = torch.tensor(
            self._paper_embedding(self.qwen_cache, str(row["ID"]), "qwen"),
            dtype=torch.float32,
        ).unsqueeze(0)
        wollm_paper = torch.tensor(
            self._paper_embedding(self.wollm_cache, str(row["ID"]), "wollm"),
            dtype=torch.float32,
        ).unsqueeze(0)

        return {
            "qwen_paper": qwen_paper,
            "wollm_paper": wollm_paper,
            "category": row["Qwen_Category_1"] if "Qwen_Category_1" in row else "unknown",
            "max_papers": self.max_reviewer_papers,
            "emb_dim": self.emb_dim,
            "task0": self._task_tensors(reviewer_ids, wrong_ids),
            "task1": self._task_tensors(reviewer_ids, similar_ids),
            "task2": self._task_tensors(similar_ids, wrong_ids),
        }


class DualPathPaperSetDataloader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def _empty_task(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        emb_dim = int(batch[0]["emb_dim"])
        max_papers = int(batch[0]["max_papers"])
        return {
            "emb_paper": torch.empty((0, emb_dim), dtype=torch.float32),
            "reviewer_paper_embs": torch.empty((0, max_papers, emb_dim), dtype=torch.float32),
            "reviewer_paper_mask": torch.empty((0, max_papers), dtype=torch.float32),
            "emb_wollm_paper": torch.empty((0, emb_dim), dtype=torch.float32),
            "reviewer_wollm_paper_embs": torch.empty((0, max_papers, emb_dim), dtype=torch.float32),
            "reviewer_wollm_paper_mask": torch.empty((0, max_papers), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.float32),
            "group_sizes": torch.empty((0,), dtype=torch.long),
            "categories": [],
        }

    @classmethod
    def _collate_task(cls, batch: list[Dict[str, torch.Tensor]], task_name: str) -> Dict[str, torch.Tensor]:
        emb_paper = []
        reviewer_paper_embs = []
        reviewer_paper_mask = []
        emb_wollm_paper = []
        reviewer_wollm_paper_embs = []
        reviewer_wollm_paper_mask = []
        labels = []
        group_sizes = []
        categories = []
        for item in batch:
            task = item[task_name]
            group_size = task["labels"].numel()
            if group_size == 0:
                continue
            emb_paper.append(item["qwen_paper"].expand(group_size, -1))
            reviewer_paper_embs.append(task["qwen_reviewers"])
            reviewer_paper_mask.append(task["qwen_masks"])
            emb_wollm_paper.append(item["wollm_paper"].expand(group_size, -1))
            reviewer_wollm_paper_embs.append(task["wollm_reviewers"])
            reviewer_wollm_paper_mask.append(task["wollm_masks"])
            labels.append(task["labels"])
            group_sizes.append(group_size)
            categories.append(item["category"])

        if not labels:
            return cls._empty_task(batch)

        return {
            "emb_paper": torch.cat(emb_paper, dim=0),
            "reviewer_paper_embs": torch.cat(reviewer_paper_embs, dim=0),
            "reviewer_paper_mask": torch.cat(reviewer_paper_mask, dim=0),
            "emb_wollm_paper": torch.cat(emb_wollm_paper, dim=0),
            "reviewer_wollm_paper_embs": torch.cat(reviewer_wollm_paper_embs, dim=0),
            "reviewer_wollm_paper_mask": torch.cat(reviewer_wollm_paper_mask, dim=0),
            "labels": torch.cat(labels, dim=0),
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
            "categories": categories,
        }

    @classmethod
    def collate_fn(cls, batch: list[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "task0": cls._collate_task(batch, "task0"),
            "task1": cls._collate_task(batch, "task1"),
            "task2": cls._collate_task(batch, "task2"),
        }


def create_dataloader(
    df_path: str,
    author_papers_path: str,
    paper_cache_path: str,
    wollm_cache_path: str,
    split: str,
    args: argparse.Namespace,
    shuffle: bool,
) -> DualPathPaperSetDataloader:
    dataset = DualPathPaperSetDataset(
        df_path=df_path,
        author_papers_path=author_papers_path,
        qwen_cache_path=paper_cache_path,
        wollm_cache_path=wollm_cache_path,
        split=split,
        args=args,
    )
    return DualPathPaperSetDataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def build_model(args: argparse.Namespace) -> nn.Module:
    return DualPathPaperSetMMoE(
        emb_dim=args.emb_dim,
        qwen_model_dim=args.qwen_model_dim,
        max_papers=args.max_reviewer_papers,
        reviewer_encoder_type=args.reviewer_encoder,
        num_transformer_layers=args.num_transformer_layers,
        num_attention_heads=args.num_attention_heads,
        fusion_hidden_dims=training_utils.parse_int_list(args.fusion_hidden_dims),
        num_tasks=3,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        expert_hidden_dims=training_utils.parse_int_list(args.expert_hidden_dims),
        tower_hidden_dims=training_utils.parse_int_list(args.tower_hidden_dims),
        dropout=args.dropout,
        router_noise=args.router_noise,
        topk=args.topk,
    )


def move_batch_to_device(
    batch: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        task_name: {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in task_batch.items()
        }
        for task_name, task_batch in batch.items()
    }


def dataset_debug_summary(dataloader: DataLoader) -> str:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return "dataset summary unavailable"
    summary_fn = getattr(dataset, "summary", None)
    if callable(summary_fn):
        return summary_fn()
    return f"dataset={type(dataset).__name__}"


def predict_outputs(
    model: nn.Module,
    task_batch: Dict[str, torch.Tensor],
    task_idx: int,
) -> Dict[str, torch.Tensor] | None:
    if task_batch["labels"].numel() == 0:
        return None
    return model(
        emb_paper=task_batch["emb_paper"],
        reviewer_paper_embs=task_batch["reviewer_paper_embs"],
        reviewer_paper_mask=task_batch["reviewer_paper_mask"],
        emb_wollm_paper=task_batch["emb_wollm_paper"],
        wollm_reviewer_paper_embs=task_batch["reviewer_wollm_paper_embs"],
        wollm_reviewer_paper_mask=task_batch["reviewer_wollm_paper_mask"],
        task_idx=task_idx,
    )


def gate_regularization(
    outputs: Sequence[Dict[str, torch.Tensor]],
    entropy_weight: float,
    load_balance_weight: float,
) -> tuple[torch.Tensor | None, Dict[str, float]]:
    if not outputs or (entropy_weight <= 0 and load_balance_weight <= 0):
        return None, {"gate_entropy": 0.0, "gate_load_balance_loss": 0.0}

    entropy_terms = []
    load_balance_terms = []
    for output in outputs:
        candidate_gate_weights = []

        gate_dense_weights = output.get("gate_dense_weights")
        if gate_dense_weights is not None:
            candidate_gate_weights.append(gate_dense_weights)

        gate_routed_weights = output.get("gate_full_weights")
        if gate_routed_weights is None:
            gate_routed_weights = output.get("gate_weights")
        if gate_routed_weights is not None:
            candidate_gate_weights.append(gate_routed_weights)

        if not candidate_gate_weights:
            continue

        for gate_weights in candidate_gate_weights:
            entropy_terms.append(
                -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
            )
            load_balance_terms.append(gate_weights.mean(dim=0).std(unbiased=False))

    if not entropy_terms:
        return None, {"gate_entropy": 0.0, "gate_load_balance_loss": 0.0}

    gate_entropy = torch.stack(entropy_terms).mean()
    gate_load_balance_loss = torch.stack(load_balance_terms).mean()
    gate_loss = -entropy_weight * gate_entropy + load_balance_weight * gate_load_balance_loss
    return gate_loss, {
        "gate_entropy": float(gate_entropy.detach().cpu().item()),
        "gate_load_balance_loss": float(gate_load_balance_loss.detach().cpu().item()),
    }


def compute_losses(
    model: nn.Module,
    batch: Dict[str, Dict[str, torch.Tensor]],
    bce_loss: BinaryLogitLoss,
    rank_loss: QueryPairwiseRankLoss,
    args: argparse.Namespace,
    active_tasks: Sequence[int] = (0, 1, 2),
) -> tuple[torch.Tensor | None, Dict[str, float]]:
    loss_terms = []
    metrics: Dict[str, float] = {}
    gate_outputs: list[Dict[str, torch.Tensor]] = []

    # task0 is a confidence classifier: true reviewers vs. wrong candidates.
    if 0 in active_tasks and batch["task0"]["labels"].numel() > 0:
        task0_outputs = predict_outputs(model, batch["task0"], 0)
        assert task0_outputs is not None
        task0_logits = task0_outputs["logits"]
        task0_labels = batch["task0"]["labels"]
        sample_weight = None
        hard_neg_weight_mean = 1.0
        if args.hard_negative_alpha > 0:
            task0_probs = torch.sigmoid(task0_logits.detach())
            sample_weight = torch.ones_like(task0_labels, dtype=task0_logits.dtype)
            wrong_mask = task0_labels <= 0.5
            if wrong_mask.any():
                sample_weight[wrong_mask] = (
                    1.0
                    + args.hard_negative_alpha
                    * task0_probs[wrong_mask].pow(args.hard_negative_gamma)
                )
                hard_neg_weight_mean = float(
                    sample_weight[wrong_mask].detach().cpu().mean().item()
                )
        loss0 = bce_loss(task0_logits, task0_labels, weight=sample_weight)
        loss_terms.append(args.lambda_task0 * loss0)
        metrics["task1_conf_loss"] = float(loss0.detach().cpu().item())
        if args.hard_negative_alpha > 0:
            metrics["task1_hard_neg_weight_mean"] = hard_neg_weight_mean
        gate_outputs.append(task0_outputs)

    # task1 ranks true reviewers above similar candidates.
    if 1 in active_tasks and batch["task1"]["labels"].numel() > 0:
        task1_outputs = predict_outputs(model, batch["task1"], 1)
        assert task1_outputs is not None
        loss1 = rank_loss(task1_outputs["logits"], batch["task1"]["labels"], batch["task1"]["group_sizes"])
        loss_terms.append(args.lambda_task1 * loss1)
        metrics["task2_rank_loss"] = float(loss1.detach().cpu().item())
        gate_outputs.append(task1_outputs)

    # task2 ranks similar candidates above clearly wrong candidates.
    if 2 in active_tasks and batch["task2"]["labels"].numel() > 0:
        task2_outputs = predict_outputs(model, batch["task2"], 2)
        assert task2_outputs is not None
        loss2 = rank_loss(task2_outputs["logits"], batch["task2"]["labels"], batch["task2"]["group_sizes"])
        loss_terms.append(args.lambda_task2 * loss2)
        metrics["task3_rank_loss"] = float(loss2.detach().cpu().item())
        gate_outputs.append(task2_outputs)

    if not loss_terms:
        return None, {}

    total_loss = torch.stack(loss_terms).sum()
    gate_loss, gate_metrics = gate_regularization(
        gate_outputs,
        entropy_weight=args.gate_entropy_weight,
        load_balance_weight=args.gate_load_balance_weight,
    )
    if gate_loss is not None:
        total_loss = total_loss + gate_loss

    metrics.update(gate_metrics)
    metrics = {"loss": float(total_loss.detach().cpu().item()), **metrics}
    return total_loss, metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    args: argparse.Namespace,
    max_steps: int | None,
    active_tasks: Sequence[int] = (0, 1, 2),
    desc: str = "Training",
) -> Dict[str, float]:
    model.train()
    bce_loss = BinaryLogitLoss()
    rank_loss = QueryPairwiseRankLoss()
    history = []

    pbar = tqdm(dataloader, desc=desc)
    task_example_counts = {"task0": 0, "task1": 0, "task2": 0}
    for step, batch in enumerate(pbar, start=1):
        task_example_counts["task0"] += int(batch["task0"]["labels"].numel())
        task_example_counts["task1"] += int(batch["task1"]["labels"].numel())
        task_example_counts["task2"] += int(batch["task2"]["labels"].numel())
        batch = move_batch_to_device(batch, device)
        loss, metrics = compute_losses(
            model=model,
            batch=batch,
            bce_loss=bce_loss,
            rank_loss=rank_loss,
            args=args,
            active_tasks=active_tasks,
        )
        if loss is None:
            if max_steps is not None and step >= max_steps:
                break
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            max_norm=args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()

        history.append(metrics)
        pbar.set_postfix({key: f"{value:.4f}" for key, value in metrics.items()})

        if max_steps is not None and step >= max_steps:
            break

    if not history:
        raise RuntimeError(
            "No valid training batches were processed. "
            f"task_example_counts={task_example_counts}. "
            f"{dataset_debug_summary(dataloader)}"
        )
    return training_utils.average_metrics(history)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    max_steps: int | None = None,
    print_mmoe_format: bool = False,
) -> Dict[str, float]:
    model.eval()
    task1_conf_correct = 0
    task1_conf_count = 0
    true_probs = []
    wrong_probs = []
    task2_metric_values = {"map": [], "Rprec": [], "recip_rank": [], "ndcg": []}
    task3_metric_values = {"map": [], "Rprec": [], "recip_rank": [], "ndcg": []}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)

            task0_outputs = predict_outputs(model, batch["task0"], 0)
            task1_outputs = predict_outputs(model, batch["task1"], 1)
            task2_outputs = predict_outputs(model, batch["task2"], 2)

            if task0_outputs is not None:
                task0_logits = task0_outputs["logits"]
                task0_labels = batch["task0"]["labels"]
                task0_probs = torch.sigmoid(task0_logits)
                task0_pred = (task0_probs >= 0.5).to(task0_labels.dtype)
                task1_conf_correct += int((task0_pred == task0_labels).sum().item())
                task1_conf_count += int(task0_labels.numel())
                true_mask = task0_labels > 0.5
                wrong_mask = task0_labels <= 0.5
                if true_mask.any():
                    true_probs.extend(task0_probs[true_mask].detach().cpu().tolist())
                if wrong_mask.any():
                    wrong_probs.extend(task0_probs[wrong_mask].detach().cpu().tolist())
                if print_mmoe_format:
                    training_utils.print_confidence_scores(
                        batch["task0"],
                        task0_logits,
                        task0_outputs.get("gate_weights"),
                    )

            if task1_outputs is not None:
                task1_metrics = training_utils.grouped_ranking_metrics(
                    task1_outputs["logits"],
                    batch["task1"]["labels"],
                    batch["task1"]["group_sizes"],
                )
                for metric_name, metric_values in task1_metrics.items():
                    task2_metric_values[metric_name].extend(metric_values)
                if print_mmoe_format:
                    training_utils.print_rank_scores(
                        batch["task1"],
                        task1_outputs["logits"],
                        "True Reviewerin Similar",
                        "Similar Reviewer",
                    )

            if task2_outputs is not None:
                task2_metrics = training_utils.grouped_ranking_metrics(
                    task2_outputs["logits"],
                    batch["task2"]["labels"],
                    batch["task2"]["group_sizes"],
                )
                for metric_name, metric_values in task2_metrics.items():
                    task3_metric_values[metric_name].extend(metric_values)
                if print_mmoe_format:
                    training_utils.print_rank_scores(
                        batch["task2"],
                        task2_outputs["logits"],
                        "Similar Reviewer",
                        "Wrong Candidates",
                    )

            if max_steps is not None and step >= max_steps:
                break

    if task1_conf_count == 0:
        raise RuntimeError(
            "No confidence evaluation examples were processed. "
            f"{dataset_debug_summary(dataloader)}"
        )

    def mean_or_zero(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    metrics = {
        "task1_conf_acc": float(task1_conf_correct / task1_conf_count),
        "task1_true_prob_mean": mean_or_zero(true_probs),
        "task1_wrong_prob_mean": mean_or_zero(wrong_probs),
        "task2_map": mean_or_zero(task2_metric_values["map"]),
        "task2_rprec": mean_or_zero(task2_metric_values["Rprec"]),
        "task2_mrr": mean_or_zero(task2_metric_values["recip_rank"]),
        "task2_ndcg": mean_or_zero(task2_metric_values["ndcg"]),
        "task3_map": mean_or_zero(task3_metric_values["map"]),
        "task3_rprec": mean_or_zero(task3_metric_values["Rprec"]),
        "task3_mrr": mean_or_zero(task3_metric_values["recip_rank"]),
        "task3_ndcg": mean_or_zero(task3_metric_values["ndcg"]),
    }
    metrics["selection_metric"] = (
        metrics["task1_conf_acc"]
        + (1.0 - metrics["task1_wrong_prob_mean"])
        + metrics["task2_ndcg"]
        + metrics["task3_ndcg"]
    ) / 4.0
    return metrics


def configure_stage2_freeze(model: nn.Module, stage2_train_shared: bool) -> None:
    if not isinstance(model, DualPathPaperSetMMoE):
        return

    # Stage 2 focuses on ranking towers by default; shared encoders can optionally
    # remain trainable when the ranking stage needs to adapt representations.
    training_utils.set_requires_grad(model, False)

    if stage2_train_shared:
        training_utils.set_requires_grad(model.qwen_encoder, True)
        training_utils.set_requires_grad(model.wollm_encoder, True)
        training_utils.set_requires_grad(model.qwen_feature_builder, True)
        training_utils.set_requires_grad(model.wollm_feature_builder, True)
        training_utils.set_requires_grad(model.fusion_encoder, True)
        training_utils.set_requires_grad(model.experts, True)

    for task_idx in (1, 2):
        training_utils.set_requires_grad(model.gates[task_idx], True)
        training_utils.set_requires_grad(model.towers[task_idx], True)
        training_utils.set_requires_grad(model.scorers[task_idx], True)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters are available for the current stage.")
    return params


def make_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    steps_per_epoch: int,
    epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, steps_per_epoch * max(1, epochs))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def save_epoch_checkpoint(
    checkpoint_dir: str,
    filename: str,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: Dict[str, float],
    enabled: bool,
) -> str | None:
    if not enabled:
        return None

    checkpoint_path = os.path.join(checkpoint_dir, "all_checkpoints", filename)
    training_utils.save_checkpoint(checkpoint_path, model, args, epoch, metrics)
    return checkpoint_path


def run_train(args: argparse.Namespace) -> None:
    training_utils.set_seed(args.seed)
    device = training_utils.resolve_device(args.device)

    train_dataloader = create_dataloader(
        df_path=args.train_df,
        author_papers_path=args.train_author_papers,
        paper_cache_path=resolve_qwen_cache(args, "train"),
        wollm_cache_path=resolve_wollm_cache(args, "train"),
        split="train",
        args=args,
        shuffle=True,
    )
    val_dataloader = create_dataloader(
        df_path=args.val_df,
        author_papers_path=args.val_author_papers,
        paper_cache_path=resolve_qwen_cache(args, "val"),
        wollm_cache_path=resolve_wollm_cache(args, "val"),
        split="test",
        args=args,
        shuffle=False,
    )
    print(f"Train dataset summary: {dataset_debug_summary(train_dataloader)}")
    print(f"Validation dataset summary: {dataset_debug_summary(val_dataloader)}")

    model = build_model(args).to(device)
    save_dir = os.path.join(
        args.save_dir,
        f"paper_set_dual_path_mmoe_{args.reviewer_encoder}",
    )
    checkpoint_path = os.path.join(save_dir, "best_model.pth")
    eval_max_steps = args.max_eval_steps if args.max_eval_steps is not None else args.max_steps
    steps_per_epoch = training_utils.effective_steps_per_epoch(train_dataloader, args.max_steps)

    if args.mmoe_staged:
        stage1_epochs = min(args.stage1_epochs, args.epochs)
        stage2_epochs = args.epochs - stage1_epochs
        print(
            f"Paper-set dual-path MMoE staged training: reviewer_encoder={args.reviewer_encoder}, "
            f"stage1_conf_epochs={stage1_epochs}, stage2_rank_epochs={stage2_epochs}, "
            f"stage2_train_shared={args.stage2_train_shared}"
        )

        training_utils.set_all_trainable(model)
        optimizer, scheduler = make_optimizer_and_scheduler(
            model,
            args,
            steps_per_epoch=steps_per_epoch,
            epochs=stage1_epochs,
        )

        stage1_checkpoint_path = os.path.join(save_dir, "stage1_conf_model.pth")
        stage1_best_score = float("-inf")
        for epoch in range(stage1_epochs):
            print(f"\n============= Stage 1 Confidence Epoch {epoch + 1}/{stage1_epochs} =============")
            train_metrics = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                args=args,
                max_steps=args.max_steps,
                active_tasks=(0,),
                desc="Training stage1_conf",
            )
            training_utils.print_metrics("Train", train_metrics)

            val_metrics = evaluate(
                model=model,
                dataloader=val_dataloader,
                device=device,
                args=args,
                max_steps=eval_max_steps,
            )
            training_utils.print_metrics("Validation", val_metrics)

            epoch_checkpoint_path = save_epoch_checkpoint(
                checkpoint_dir=save_dir,
                filename=f"stage1_epoch_{epoch + 1:03d}.pth",
                model=model,
                args=args,
                epoch=epoch,
                metrics=val_metrics,
                enabled=args.save_all_checkpoints,
            )
            if epoch_checkpoint_path is not None:
                print(f"Saved stage1 epoch checkpoint to {epoch_checkpoint_path}")

            stage1_score = training_utils.stage1_confidence_score(val_metrics)
            if stage1_score > stage1_best_score:
                stage1_best_score = stage1_score
                training_utils.save_checkpoint(stage1_checkpoint_path, model, args, epoch, val_metrics)
                print(
                    f"New best stage1 checkpoint saved to {stage1_checkpoint_path} "
                    f"with confidence_score: {stage1_best_score:.4f}"
                )

        training_utils.load_checkpoint(model, stage1_checkpoint_path, device)
        print(f"Loaded best stage1 checkpoint: {stage1_checkpoint_path}")

        best_score = float("-inf")
        if stage2_epochs > 0:
            configure_stage2_freeze(model, stage2_train_shared=args.stage2_train_shared)
            optimizer, scheduler = make_optimizer_and_scheduler(
                model,
                args,
                steps_per_epoch=steps_per_epoch,
                epochs=stage2_epochs,
            )
            for stage2_epoch in range(stage2_epochs):
                epoch = stage1_epochs + stage2_epoch
                print(f"\n============= Stage 2 Ranking Epoch {stage2_epoch + 1}/{stage2_epochs} =============")
                train_metrics = train_epoch(
                    model=model,
                    dataloader=train_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    args=args,
                    max_steps=args.max_steps,
                    active_tasks=(1, 2),
                    desc="Training stage2_rank",
                )
                training_utils.print_metrics("Train", train_metrics)

                val_metrics = evaluate(
                    model=model,
                    dataloader=val_dataloader,
                    device=device,
                    args=args,
                    max_steps=eval_max_steps,
                )
                training_utils.print_metrics("Validation", val_metrics)

                epoch_checkpoint_path = save_epoch_checkpoint(
                    checkpoint_dir=save_dir,
                    filename=f"stage2_epoch_{stage2_epoch + 1:03d}_global_{epoch + 1:03d}.pth",
                    model=model,
                    args=args,
                    epoch=epoch,
                    metrics=val_metrics,
                    enabled=args.save_all_checkpoints,
                )
                if epoch_checkpoint_path is not None:
                    print(f"Saved stage2 epoch checkpoint to {epoch_checkpoint_path}")

                if val_metrics["selection_metric"] > best_score:
                    best_score = val_metrics["selection_metric"]
                    training_utils.save_checkpoint(checkpoint_path, model, args, epoch, val_metrics)
                    print(f"New best checkpoint saved to {checkpoint_path}")
        else:
            val_metrics = evaluate(
                model=model,
                dataloader=val_dataloader,
                device=device,
                args=args,
                max_steps=eval_max_steps,
            )
            best_score = val_metrics["selection_metric"]
            training_utils.save_checkpoint(checkpoint_path, model, args, stage1_epochs - 1, val_metrics)
            print(f"Stage2 skipped; saved stage1 model to {checkpoint_path}")

        print(f"Best stage1 confidence_score: {stage1_best_score:.4f}")
        print(f"Best validation selection_metric: {best_score:.4f}")
        print(f"Best checkpoint: {checkpoint_path}")
        return

    training_utils.set_all_trainable(model)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model,
        args,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
    )

    best_score = float("-inf")
    for epoch in range(args.epochs):
        print(f"\n============= Epoch {epoch + 1}/{args.epochs} =============")
        train_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            max_steps=args.max_steps,
            active_tasks=(0, 1, 2),
        )
        training_utils.print_metrics("Train", train_metrics)

        val_metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            args=args,
            max_steps=eval_max_steps,
        )
        training_utils.print_metrics("Validation", val_metrics)

        epoch_checkpoint_path = save_epoch_checkpoint(
            checkpoint_dir=save_dir,
            filename=f"epoch_{epoch + 1:03d}.pth",
            model=model,
            args=args,
            epoch=epoch,
            metrics=val_metrics,
            enabled=args.save_all_checkpoints,
        )
        if epoch_checkpoint_path is not None:
            print(f"Saved epoch checkpoint to {epoch_checkpoint_path}")

        if val_metrics["selection_metric"] > best_score:
            best_score = val_metrics["selection_metric"]
            training_utils.save_checkpoint(checkpoint_path, model, args, epoch, val_metrics)
            print(f"New best checkpoint saved to {checkpoint_path}")

    print(f"Best validation selection_metric: {best_score:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")


def run_test(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required when --mode test.")

    training_utils.set_seed(args.seed)
    device = training_utils.resolve_device(args.device)
    test_dataloader = create_dataloader(
        df_path=args.test_df,
        author_papers_path=args.test_author_papers,
        paper_cache_path=resolve_qwen_cache(args, "test"),
        wollm_cache_path=resolve_wollm_cache(args, "test"),
        split="test",
        args=args,
        shuffle=False,
    )
    print(f"Test dataset summary: {dataset_debug_summary(test_dataloader)}")

    model = build_model(args).to(device)
    training_utils.load_checkpoint(model, args.checkpoint, device)
    metrics = evaluate(
        model=model,
        dataloader=test_dataloader,
        device=device,
        args=args,
        max_steps=args.max_eval_steps,
        print_mmoe_format=args.print_mmoe_format,
    )
    training_utils.print_metrics("Test", metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test paper-set dual-path MMoE reviewer reranker.")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--save_all_checkpoints", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_task0", type=float, default=1.0)
    parser.add_argument("--lambda_task1", type=float, default=1.0)
    parser.add_argument("--lambda_task2", type=float, default=1.0)
    parser.add_argument("--mmoe_staged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_train_shared", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--router_noise", type=float, default=0.05)
    parser.add_argument("--gate_entropy_weight", type=float, default=0.01)
    parser.add_argument("--gate_load_balance_weight", type=float, default=0.1)
    parser.add_argument("--hard_negative_alpha", type=float, default=2.0)
    parser.add_argument("--hard_negative_gamma", type=float, default=2.0)
    parser.add_argument("--print_mmoe_format", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--qwen_model_dim", type=int, default=256)
    parser.add_argument("--max_reviewer_papers", type=int, default=10)
    parser.add_argument("--citation_topk", type=int, default=5)
    parser.add_argument("--reviewer_encoder", choices=["transformer", "mean"], default="mean")
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--fusion_hidden_dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--expert_dim", type=int, default=128)
    parser.add_argument("--expert_hidden_dims", default="512,256")
    parser.add_argument("--tower_hidden_dims", default="256,128")

    parser.add_argument("--paper_embedding_cache", default=None)
    parser.add_argument("--wollm_embedding_cache", default=None)
    parser.add_argument("--train_paper_cache", default=training_utils.DEFAULT_TRAIN_PAPER_CACHE)
    parser.add_argument("--val_paper_cache", default=training_utils.DEFAULT_VAL_PAPER_CACHE)
    parser.add_argument("--test_paper_cache", default=training_utils.DEFAULT_TEST_PAPER_CACHE)
    parser.add_argument("--train_wollm_cache", default=training_utils.DEFAULT_TRAIN_WOLLM_CACHE)
    parser.add_argument("--val_wollm_cache", default=training_utils.DEFAULT_VAL_WOLLM_CACHE)
    parser.add_argument("--test_wollm_cache", default=training_utils.DEFAULT_TEST_WOLLM_CACHE)

    parser.add_argument("--train_df", default=training_utils.DEFAULT_TRAIN_DF)
    parser.add_argument("--val_df", default=training_utils.DEFAULT_VAL_DF)
    parser.add_argument("--test_df", default=training_utils.DEFAULT_TEST_DF)
    parser.add_argument("--train_author_papers", default=training_utils.DEFAULT_TRAIN_AUTHOR_PAPERS)
    parser.add_argument("--val_author_papers", default=training_utils.DEFAULT_VAL_AUTHOR_PAPERS)
    parser.add_argument("--test_author_papers", default=training_utils.DEFAULT_TEST_AUTHOR_PAPERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_test(args)


if __name__ == "__main__":
    main()
