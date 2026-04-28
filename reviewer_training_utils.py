import argparse
import json
import math
import os
import random
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_DIR = os.path.join(BASE_DIR, "data/checkpoints")

# Keep all public defaults repository-relative to avoid leaking local machine paths.
DEFAULT_TRAIN_DF = "data/train_dataset_qwen_sampled.parquet"
DEFAULT_VAL_DF = "data/val_dataset_qwen_sampled.parquet"
DEFAULT_TEST_DF = "data/test_dataset_qwen_sampled.parquet"

DEFAULT_TRAIN_AUTHOR_PAPERS = "data/cache/train_sampled_author_papers.jsonl"
DEFAULT_VAL_AUTHOR_PAPERS = "data/cache/val_sampled_author_papers.jsonl"
DEFAULT_TEST_AUTHOR_PAPERS = "data/cache/test_sampled_author_papers.jsonl"

DEFAULT_TRAIN_PAPER_CACHE = "data/cache/train_only_paper_qwen_summary_embedding.jsonl"
DEFAULT_VAL_PAPER_CACHE = "data/cache/val_only_paper_qwen_summary_embedding.jsonl"
DEFAULT_TEST_PAPER_CACHE = "data/cache/test_only_paper_qwen_summary_embedding.jsonl"

DEFAULT_TRAIN_WOLLM_CACHE = "data/cache/train_wollm_qwen_summary_embedding.jsonl"
DEFAULT_VAL_WOLLM_CACHE = "data/cache/val_wollm_qwen_summary_embedding.jsonl"
DEFAULT_TEST_WOLLM_CACHE = "data/cache/test_wollm_qwen_summary_embedding.jsonl"


def parse_int_list(value: str) -> tuple[int, ...]:
    if value.strip() == "":
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device {device_name}, but CUDA is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(device_name)


def to_list(value) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = list(value)
    return [str(item) for item in value]


def load_author_publications(path: str) -> dict[str, list[dict]]:
    """Load reviewer publication lists keyed by author id."""
    author_papers: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"Loading {os.path.basename(path)}"):
            item = json.loads(line)
            author_papers[str(item["author_id"])] = list(item.get("publications", []))
    return author_papers


def select_candidate_papers(
    publications: Sequence[dict],
    topk: int = 10,
    citation_topk: int = 5,
) -> list[dict]:
    """Prefer cited papers first, then fill the reviewer profile with recent papers."""
    publications = list(publications)
    if len(publications) <= topk:
        return publications

    cited = [pub for pub in publications if pub.get("n_citation", 0) > 0]
    cited.sort(key=lambda pub: pub.get("n_citation", 0), reverse=True)
    if len(cited) >= citation_topk:
        selected = cited[:citation_topk]
        left = topk - citation_topk
    else:
        selected = cited
        left = topk - len(cited)

    if left > 0:
        selected_ids = {pub.get("id") for pub in selected}
        remaining = [pub for pub in publications if pub.get("id") not in selected_ids]
        remaining.sort(key=lambda pub: pub.get("year", 0) or 0, reverse=True)
        selected.extend(remaining[:left])

    return selected[:topk]


def sample_dataframe_indices(
    df: pd.DataFrame,
    sample_ratio: float,
    seed: int,
) -> list[int]:
    """Sample rows globally or per qwen category so small experiments keep label coverage."""
    if sample_ratio >= 1.0:
        return list(range(len(df)))
    if sample_ratio <= 0:
        raise ValueError(f"sample_ratio must be > 0, got {sample_ratio}.")

    sampled_indices: list[int] = []
    if "Qwen_Category_1" not in df:
        sample_size = min(len(df), max(1, math.ceil(len(df) * sample_ratio)))
        return sorted(df.sample(n=sample_size, random_state=seed).index.tolist())

    for _, group in df.groupby("Qwen_Category_1", group_keys=False):
        group_size = len(group)
        sample_size = min(group_size, max(1, math.ceil(group_size * sample_ratio)))
        sampled_indices.extend(group.sample(n=sample_size, random_state=seed).index.tolist())
    return sorted(sampled_indices)


def sample_candidates(
    source_idx: int,
    reviewer_count: int,
    wrong_candidates: Sequence[str],
    similar_candidates: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Use the row index as a stable seed so sampled negatives are reproducible."""
    rng = random.Random(source_idx)
    wrong_candidates = list(wrong_candidates)
    similar_candidates = list(similar_candidates)
    wrong_count = min(reviewer_count, len(wrong_candidates))
    similar_count = min(reviewer_count, len(similar_candidates))
    wrong_ids = rng.sample(wrong_candidates, wrong_count) if wrong_count > 0 else []
    similar_ids = rng.sample(similar_candidates, similar_count) if similar_count > 0 else []
    return wrong_ids, similar_ids


def row_candidate_ids(
    row: pd.Series,
    source_idx: int,
    use_full_pool: bool,
) -> tuple[list[str], list[str], list[str]]:
    reviewer_ids = to_list(row["Reviewer_IDs"])
    if use_full_pool:
        wrong_ids = to_list(row["Wrong_Candidates"])
        similar_ids = to_list(row["Similar_Candidates"])
    else:
        wrong_ids, similar_ids = sample_candidates(
            source_idx,
            len(reviewer_ids),
            to_list(row["Wrong_Candidates"]),
            to_list(row["Similar_Candidates"]),
        )
    return reviewer_ids, wrong_ids, similar_ids


def paper_cache_key(paper_id: str) -> str:
    paper_id = str(paper_id)
    return paper_id if paper_id.startswith("paper:") else f"paper:{paper_id}"


def collect_required_paper_ids(
    df: pd.DataFrame,
    indices: Sequence[int],
    author_publications: dict[str, list[dict]],
    use_full_pool: bool,
    max_papers: int,
    citation_topk: int,
) -> set[str]:
    """Collect target and reviewer-paper ids before streaming large embedding caches."""
    required: set[str] = set()
    missing_authors: set[str] = set()
    for source_idx in tqdm(indices, desc="Collecting required paper ids"):
        row = df.iloc[source_idx]
        required.add(str(row["ID"]))
        reviewer_ids, wrong_ids, similar_ids = row_candidate_ids(row, source_idx, use_full_pool)
        for author_id in reviewer_ids + wrong_ids + similar_ids:
            if str(author_id) not in author_publications:
                missing_authors.add(str(author_id))
                continue
            for paper in select_candidate_papers(
                author_publications.get(str(author_id), []),
                topk=max_papers,
                citation_topk=citation_topk,
            ):
                if paper.get("id") is not None:
                    required.add(str(paper["id"]))
    if missing_authors:
        examples = sorted(missing_authors)[:10]
        print(f"Missing author publication examples: {examples}; total={len(missing_authors)}")
    return required


def load_filtered_paper_cache(
    cache_path: str,
    required_paper_ids: set[str],
    emb_dim: int,
) -> dict[str, np.ndarray]:
    """Load only embeddings needed by the active split to keep memory usage bounded."""
    cache: dict[str, np.ndarray] = {}
    required_keys = {paper_cache_key(paper_id) for paper_id in required_paper_ids}
    with open(cache_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"Loading filtered {os.path.basename(cache_path)}"):
            item = json.loads(line)
            item_id = str(item.get("id", ""))
            if item_id not in required_keys:
                continue
            embedding = np.asarray(item["embedding"], dtype=np.float32)
            if embedding.shape != (emb_dim,):
                raise ValueError(
                    f"Embedding dim mismatch for {item_id}: expected {emb_dim}, got {embedding.shape}."
                )
            cache[item_id] = embedding

    missing = len(required_keys - set(cache.keys()))
    print(
        f"paper cache loaded: required={len(required_keys)}, loaded={len(cache)}, missing={missing}"
    )
    if missing > 0:
        examples = sorted(required_keys - set(cache.keys()))[:10]
        print(f"Missing paper embedding examples: {examples}")
    return cache

def ndcg(scores: Sequence[float], labels: Sequence[float]) -> float:
    if len(scores) == 0:
        return 0.0
    ranked_labels = [
        label for _, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    ]
    ideal_labels = sorted(labels, reverse=True)

    def dcg(relevance: Sequence[float]) -> float:
        value = 0.0
        for rank, rel in enumerate(relevance, start=1):
            value += (2.0 ** float(rel) - 1.0) / np.log2(rank + 1)
        return value

    ideal_dcg = dcg(ideal_labels)
    if ideal_dcg <= 0:
        return 0.0
    return float(dcg(ranked_labels) / ideal_dcg)


def query_ndcg(scores: Sequence[float], labels: Sequence[float], query_id: str) -> float:
    if pytrec_eval is None:
        return ndcg(scores, labels)

    qrel = {query_id: {}}
    run = {query_id: {}}
    for idx, (score, label) in enumerate(zip(scores, labels), start=1):
        doc_id = str(idx)
        relevance = int(label)
        if relevance > 0:
            qrel[query_id][doc_id] = relevance
        run[query_id][doc_id] = float(score)

    if not qrel[query_id]:
        return 0.0

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"ndcg"})
    results = evaluator.evaluate(run)
    return float(results[query_id]["ndcg"])


def average_precision(scores: Sequence[float], labels: Sequence[float]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hit_count = 0
    precision_sum = 0.0
    positive_count = sum(1 for _, label in ranked if label > 0)
    if positive_count == 0:
        return 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label > 0:
            hit_count += 1
            precision_sum += hit_count / rank
    return float(precision_sum / positive_count)


def reciprocal_rank(scores: Sequence[float], labels: Sequence[float]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    for rank, (_, label) in enumerate(ranked, start=1):
        if label > 0:
            return float(1.0 / rank)
    return 0.0


def r_precision(scores: Sequence[float], labels: Sequence[float]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positive_count = sum(1 for _, label in ranked if label > 0)
    if positive_count == 0:
        return 0.0
    top_r = ranked[:positive_count]
    hits = sum(1 for _, label in top_r if label > 0)
    return float(hits / positive_count)


def evaluate_candidate_run(
    qrel: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    metrics: set[str],
) -> dict[str, float]:
    query_id = next(iter(run))
    if pytrec_eval is not None:
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
        return {
            metric: float(value)
            for metric, value in evaluator.evaluate(run)[query_id].items()
        }

    labels = [float(qrel.get(query_id, {}).get(doc_id, 0)) for doc_id in run[query_id]]
    scores = [float(score) for score in run[query_id].values()]
    fallback = {
        "map": average_precision(scores, labels),
        "Rprec": r_precision(scores, labels),
        "recip_rank": reciprocal_rank(scores, labels),
        "ndcg": ndcg(scores, labels),
    }
    return {metric: fallback[metric] for metric in metrics if metric in fallback}


def grouped_ranking_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
) -> dict[str, list[float]]:
    values = {"map": [], "Rprec": [], "recip_rank": [], "ndcg": []}
    if scores.numel() == 0:
        return values

    scores_list = scores.detach().cpu().reshape(-1).tolist()
    labels_list = labels.detach().cpu().reshape(-1).tolist()
    group_sizes_list = [int(size) for size in group_sizes.detach().cpu().tolist()]

    offset = 0
    for group_idx, group_size in enumerate(group_sizes_list):
        group_scores = scores_list[offset : offset + group_size]
        group_labels = labels_list[offset : offset + group_size]
        query_id = str(group_idx)
        qrel = {query_id: {}}
        run = {query_id: {}}
        for local_idx, (score, label) in enumerate(zip(group_scores, group_labels), start=1):
            doc_id = str(local_idx)
            if label > 0:
                qrel[query_id][doc_id] = int(label)
            run[query_id][doc_id] = float(score)

        if qrel[query_id]:
            query_metrics = evaluate_candidate_run(
                qrel,
                run,
                {"map", "Rprec", "recip_rank", "ndcg"},
            )
        else:
            query_metrics = {"map": 0.0, "Rprec": 0.0, "recip_rank": 0.0, "ndcg": 0.0}

        for metric_name in values:
            values[metric_name].append(float(query_metrics.get(metric_name, 0.0)))
        offset += group_size

    return values


def grouped_ndcg(
    scores: torch.Tensor,
    labels: torch.Tensor,
    group_sizes: torch.Tensor,
) -> float:
    if scores.numel() == 0:
        return 0.0
    scores_list = scores.detach().cpu().reshape(-1).tolist()
    labels_list = labels.detach().cpu().reshape(-1).tolist()
    group_sizes_list = [int(size) for size in group_sizes.detach().cpu().tolist()]

    values = []
    offset = 0
    for group_size in group_sizes_list:
        values.append(
            query_ndcg(
                scores_list[offset : offset + group_size],
                labels_list[offset : offset + group_size],
                str(len(values)),
            )
        )
        offset += group_size
    return float(np.mean(np.asarray(values))) if values else 0.0


def average_metrics(history: list[Dict[str, float]]) -> Dict[str, float]:
    keys = history[0].keys()
    return {
        key: float(np.mean(np.asarray([metrics[key] for metrics in history])))
        for key in keys
    }

def print_confidence_scores(
    task_batch: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    gate_weights: torch.Tensor | None = None,
) -> None:
    probs = torch.sigmoid(logits).detach().cpu().reshape(-1).tolist()
    labels = task_batch["labels"].detach().cpu().reshape(-1).tolist()
    categories = task_batch.get("categories", [])
    group_sizes = [int(size) for size in task_batch["group_sizes"].detach().cpu().tolist()]
    offset = 0
    for group_idx, group_size in enumerate(group_sizes):
        category = categories[group_idx] if group_idx < len(categories) else "unknown"
        end = offset + group_size
        if gate_weights is not None:
            print(gate_weights[offset:end].detach().cpu())
        for score, label in zip(probs[offset:end], labels[offset:end]):
            if label > 0.5:
                print(f"Final Cosine Similarity Score (True Reviewer): {score:.4f}  Category: {category}")
            else:
                print(f"Final Cosine Similarity Score (Wrong Candidates): {score:.4f}  Category: {category}")
        offset = end


def print_rank_scores(
    task_batch: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    positive_name: str,
    negative_name: str,
) -> None:
    scores = torch.sigmoid(logits).detach().cpu().reshape(-1).tolist()
    labels = task_batch["labels"].detach().cpu().reshape(-1).tolist()
    categories = task_batch.get("categories", [])
    group_sizes = [int(size) for size in task_batch["group_sizes"].detach().cpu().tolist()]
    offset = 0
    for group_idx, group_size in enumerate(group_sizes):
        category = categories[group_idx] if group_idx < len(categories) else "unknown"
        end = offset + group_size
        candidates = []
        qrel = {str(group_idx): {}}
        run = {str(group_idx): {}}
        for local_idx, (score, label) in enumerate(zip(scores[offset:end], labels[offset:end]), start=1):
            if label > 0.5:
                print(f"Final Cosine Similarity Score ({positive_name}): {score:.4f}  Category: {category}")
                qrel[str(group_idx)][str(local_idx)] = 1
            else:
                print(f"Final Cosine Similarity Score ({negative_name}): {score:.4f}  Category: {category}")
            candidates.append((local_idx, score))
        print(candidates)
        print(f"All Candidates before sorting: {len(candidates)}")
        candidates.sort(key=lambda item: item[1], reverse=True)
        print(candidates)
        run[str(group_idx)] = {str(cid): float(score) for cid, score in candidates}
        print(run)
        print(qrel)
        if qrel[str(group_idx)]:
            results = evaluate_candidate_run(
                qrel,
                run,
                {"ndcg", "map", "recip_rank", "success_5", "Rprec"},
            )
            print(f"Evaluation Metrics: {results}")
        offset = end


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    formatted = "  ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
    print(f"{prefix} {formatted}")

def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def set_all_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True

def effective_steps_per_epoch(dataloader: DataLoader, max_steps: int | None) -> int:
    return len(dataloader) if max_steps is None else min(len(dataloader), max_steps)


def stage1_confidence_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["task1_true_prob_mean"]
        + (1.0 - metrics["task1_wrong_prob_mean"])
    ) / 2.0
