import argparse
import json
import os
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None


RANKING_METRICS = {"map", "Rprec", "recip_rank", "ndcg", "success_5"}


def to_list(value) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        value = list(value)
    return [str(item) for item in value]


def paper_cache_key(paper_id: str) -> str:
    paper_id = str(paper_id)
    return paper_id if paper_id.startswith("paper:") else f"paper:{paper_id}"


def load_author_publications(path: str) -> dict[str, list[dict]]:
    author_papers: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"Loading {os.path.basename(path)}"):
            item = json.loads(line)
            author_papers[str(item["author_id"])] = list(item.get("publications", []))
    return author_papers


def load_embedding_cache(path: str, emb_dim: int | None = None) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc=f"Loading {os.path.basename(path)}"):
            item = json.loads(line)
            embedding = np.asarray(item["embedding"], dtype=np.float32)
            if emb_dim is not None and embedding.shape != (emb_dim,):
                raise ValueError(
                    f"Embedding dim mismatch for {item['id']}: expected {emb_dim}, got {embedding.shape}."
                )
            cache[str(item["id"])] = embedding
    return cache


def select_candidate_papers(
    publications: Sequence[dict],
    topk: int,
    citation_topk: int,
) -> list[dict]:
    publications = list(publications)
    if len(publications) <= topk:
        return publications

    cited = [pub for pub in publications if pub.get("n_citation", 0) > 0]
    cited.sort(key=lambda pub: pub.get("n_citation", 0), reverse=True)
    selected = cited[: min(citation_topk, len(cited))]

    left = topk - len(selected)
    if left > 0:
        selected_ids = {pub.get("id") for pub in selected}
        remaining = [pub for pub in publications if pub.get("id") not in selected_ids]
        remaining.sort(key=lambda pub: pub.get("year", 0) or 0, reverse=True)
        selected.extend(remaining[:left])

    return selected[:topk]


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector
    return vector / norm


def mean_reviewer_score(
    paper_embedding: np.ndarray,
    publications: Sequence[dict],
    embedding_cache: dict[str, np.ndarray],
    max_papers: int,
    citation_topk: int,
) -> float | None:
    selected = select_candidate_papers(publications, topk=max_papers, citation_topk=citation_topk)
    paper_embedding = normalize(paper_embedding)
    scores = []
    for paper in selected:
        key = paper_cache_key(str(paper.get("id", "")))
        if key not in embedding_cache:
            continue
        candidate_embedding = normalize(embedding_cache[key])
        scores.append(float(candidate_embedding @ paper_embedding))
    if not scores:
        return None
    return float(np.mean(np.asarray(scores, dtype=np.float32)))


def maybe_calibrate(score: float, calibrator) -> float:
    if calibrator is None:
        return score
    try:
        transformed = calibrator.transform([score])
    except ValueError:
        transformed = calibrator.transform([[score]])
    return float(np.asarray(transformed).reshape(-1)[0])


def fallback_ranking_metrics(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positive_count = sum(1 for _, label in ranked if label > 0)
    if positive_count == 0:
        return {metric: 0.0 for metric in RANKING_METRICS}

    hit_count = 0
    precision_sum = 0.0
    reciprocal_rank = 0.0
    success_5 = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label > 0:
            hit_count += 1
            precision_sum += hit_count / rank
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
            if rank <= 5:
                success_5 = 1.0

    top_r = ranked[:positive_count]
    rprec = sum(1 for _, label in top_r if label > 0) / positive_count

    def dcg(values: Sequence[int]) -> float:
        return float(
            sum((2.0 ** float(rel) - 1.0) / np.log2(rank + 1) for rank, rel in enumerate(values, start=1))
        )

    ranked_labels = [label for _, label in ranked]
    ideal_labels = sorted(labels, reverse=True)
    ideal_dcg = dcg(ideal_labels)
    ndcg = 0.0 if ideal_dcg <= 0 else dcg(ranked_labels) / ideal_dcg
    return {
        "map": float(precision_sum / positive_count),
        "Rprec": float(rprec),
        "recip_rank": float(reciprocal_rank),
        "ndcg": float(ndcg),
        "success_5": float(success_5),
    }


def ranking_metrics(query_id: str, positive_scores: list[tuple[str, float]], negative_scores: list[tuple[str, float]]) -> dict[str, float]:
    if not positive_scores or not negative_scores:
        return {metric: 0.0 for metric in RANKING_METRICS}

    if pytrec_eval is None:
        labels = [1] * len(positive_scores) + [0] * len(negative_scores)
        scores = [score for _, score in positive_scores + negative_scores]
        return fallback_ranking_metrics(scores, labels)

    qrel = {query_id: {}}
    run = {query_id: {}}
    for idx, (candidate_id, score) in enumerate(positive_scores, start=1):
        doc_id = f"pos:{idx}:{candidate_id}"
        qrel[query_id][doc_id] = 1
        run[query_id][doc_id] = float(score)
    for idx, (candidate_id, score) in enumerate(negative_scores, start=1):
        run[query_id][f"neg:{idx}:{candidate_id}"] = float(score)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, RANKING_METRICS)
    return {metric: float(value) for metric, value in evaluator.evaluate(run)[query_id].items()}


def append_metrics(store: dict[str, list[float]], metrics: dict[str, float]) -> None:
    for metric, value in metrics.items():
        store[metric].append(float(value))


def mean_metrics(store: dict[str, list[float]], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{metric.lower()}": float(np.mean(values)) if values else 0.0
        for metric, values in sorted(store.items())
    }


def collect_scores(
    reviewer_ids: Sequence[str],
    paper_embedding: np.ndarray,
    author_publications: dict[str, list[dict]],
    embedding_cache: dict[str, np.ndarray],
    max_papers: int,
    citation_topk: int,
    calibrator,
) -> list[tuple[str, float]]:
    scored = []
    for reviewer_id in reviewer_ids:
        publications = author_publications.get(str(reviewer_id), [])
        if not publications:
            continue
        score = mean_reviewer_score(
            paper_embedding=paper_embedding,
            publications=publications,
            embedding_cache=embedding_cache,
            max_papers=max_papers,
            citation_topk=citation_topk,
        )
        if score is None:
            continue
        scored.append((str(reviewer_id), maybe_calibrate(score, calibrator)))
    return scored


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    df = pd.read_parquet(args.df)
    author_publications = load_author_publications(args.author_papers)
    embedding_cache = load_embedding_cache(args.embedding_cache, emb_dim=args.emb_dim)

    calibrator = None
    if args.calibrator:
        if joblib is None:
            raise ImportError("joblib is required when --calibrator is provided.")
        calibrator = joblib.load(args.calibrator)

    task2_true_similar = defaultdict(list)
    task3_true_wrong = defaultdict(list)
    task4_similar_wrong = defaultdict(list)
    true_scores = []
    wrong_scores = []
    skipped = 0

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating baseline"):
        paper_key = paper_cache_key(str(row["ID"]))
        if paper_key not in embedding_cache:
            skipped += 1
            continue

        paper_embedding = embedding_cache[paper_key]
        reviewer_ids = to_list(row["Reviewer_IDs"])
        wrong_ids = to_list(row["Wrong_Candidates"])
        similar_ids = to_list(row["Similar_Candidates"])

        true_scored = collect_scores(
            reviewer_ids,
            paper_embedding,
            author_publications,
            embedding_cache,
            args.max_reviewer_papers,
            args.citation_topk,
            calibrator,
        )
        wrong_scored = collect_scores(
            wrong_ids,
            paper_embedding,
            author_publications,
            embedding_cache,
            args.max_reviewer_papers,
            args.citation_topk,
            calibrator,
        )
        similar_scored = collect_scores(
            similar_ids,
            paper_embedding,
            author_publications,
            embedding_cache,
            args.max_reviewer_papers,
            args.citation_topk,
            calibrator,
        )

        true_scores.extend(score for _, score in true_scored)
        wrong_scores.extend(score for _, score in wrong_scored)

        query_id = str(row_idx)
        append_metrics(task2_true_similar, ranking_metrics(query_id, true_scored, similar_scored))
        append_metrics(task3_true_wrong, ranking_metrics(query_id, true_scored, wrong_scored))
        append_metrics(task4_similar_wrong, ranking_metrics(query_id, similar_scored, wrong_scored))

    true_arr = np.asarray(true_scores, dtype=np.float32)
    wrong_arr = np.asarray(wrong_scores, dtype=np.float32)
    conf_count = int(true_arr.size + wrong_arr.size)
    conf_correct = int((true_arr >= args.confidence_threshold).sum() + (wrong_arr < args.confidence_threshold).sum())

    results = {
        "task1_conf_acc": float(conf_correct / conf_count) if conf_count else 0.0,
        "task1_true_score_mean": float(true_arr.mean()) if true_arr.size else 0.0,
        "task1_wrong_score_mean": float(wrong_arr.mean()) if wrong_arr.size else 0.0,
        "skipped_queries": float(skipped),
    }
    results.update(mean_metrics(task2_true_similar, "task2_true_similar"))
    results.update(mean_metrics(task3_true_wrong, "task3_true_wrong"))
    results.update(mean_metrics(task4_similar_wrong, "task4_similar_wrong"))
    results["selection_metric"] = (
        results["task1_conf_acc"]
        + (1.0 - results["task1_wrong_score_mean"])
        + results["task2_true_similar_ndcg"]
        + results["task4_similar_wrong_ndcg"]
    ) / 4.0
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a generic embedding-similarity reviewer baseline.")
    parser.add_argument("--df", default="data/test_dataset_qwen_sampled.parquet")
    parser.add_argument("--author_papers", default="data/cache/test_sampled_author_papers.jsonl")
    parser.add_argument("--embedding_cache", default="data/cache/test_only_paper_qwen_summary_embedding.jsonl")
    parser.add_argument("--calibrator", default=None)
    parser.add_argument("--emb_dim", type=int, default=1024)
    parser.add_argument("--max_reviewer_papers", type=int, default=10)
    parser.add_argument("--citation_topk", type=int, default=5)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    results = evaluate(parse_args())
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
