# OmniReview

OmniReview is a reviewer-paper matching pipeline. It extracts reviewer author ids, gathers each author's papers from OAG-style metadata, summarizes papers with an LLM, embeds both LLM summaries and raw paper text, and runs a dual-path paper-set MMoE reranker for inference.

Datasets are available at [HYHPING2023/OmniReview](https://huggingface.co/datasets/HYHPING2023/OmniReview). The dataset provides OAG IDs and ORCID IDs that point to corresponding records in [Open Academic Graph](https://www.aminer.cn/open/article?id=5965cf249ed5db41ed4f52bf), using OAG version 3.2. The OAG per-paper Discipline Taxonomy files are provided in the same dataset repository. Model weights are available at [HYHPING2023/OmniReview](https://huggingface.co/HYHPING2023/OmniReview).

## Repository Layout

- `preprocess.py`: preprocessing entry points for author-id extraction, LLM summary generation, summary embedding, and raw text embedding.
- `llm/summarizer.py`: OpenAI-compatible vLLM client and prompt templates for paper and reviewer summaries.
- `models/encoders.py`: vLLM embedding client used to encode summaries and raw text.
- `train_paper_set_dual_path_mmoe.py`: training and inference entry point for the dual-path paper-set MMoE reranker.
- `reviewer_training_utils.py`: shared path defaults, data loading, candidate sampling, cache loading, metrics, and checkpoint utilities.
- `models/paper_set_dual_path_mmoe.py`: dual-path MMoE model that fuses summary and raw-text embedding paths.
- `models/reviewer_paper_encoders.py`: reviewer paper-set encoders based on masked mean pooling or Transformer encoding.
- `models/rerankers.py`: interaction feature builders and ranking/classification losses.
- `baseline/`: baseline implementations and evaluation utilities, including dual-tower, RGCN, generic embedding-similarity evaluation, and confidence calibration.
- `data/category.json`: full Discipline Taxonomy metadata.
- `data/cache/`: generated author-paper files, summaries, embeddings, and other local caches.
- `data/checkpoints/`: local checkpoints if you train or fine-tune the reranker.

## Installation

Use Python 3.10+ and install the runtime dependencies:

```bash
pip install torch numpy pandas pyarrow tqdm requests transformers vllm huggingface_hub
```

Optional ranking evaluation support:

```bash
pip install pytrec_eval
```

## vLLM Services

The preprocessing code expects OpenAI-compatible vLLM endpoints for both summarization and embedding. The paper summarization LLM is `Qwen3-30B-A3B`, and the embedding model is `Qwen3-Embedding-4B`.

Start the LLM service used by `llm/summarizer.py`:

```bash
vllm serve Qwen/Qwen3-30B-A3B --port 20000
```

Start the embedding service used by `models/encoders.py`:

```bash
vllm serve Qwen/Qwen3-Embedding-4B --port 9002 --task embed
```

If you use different ports or model names, update `LLM_PORTS` and `LLM_HOST` in `llm/summarizer.py`, and update the embedding endpoint mapping in `models/encoders.py`.

## Preprocessing Pipeline

The preprocessing workflow is implemented in `preprocess.py`. It first extracts reviewer author ids from the train, validation, and test parquet files. Those author ids are then used to retrieve each author's paper list from OAG-style metadata, producing author-to-publication JSONL files for each split. The OAG per-paper Discipline Taxonomy used by this process is available together with the dataset on Hugging Face, while the complete Discipline Taxonomy metadata is kept under `data/`.

After author papers are prepared, `preprocess.py` sends paper ids and paper content to the LLM summarization service and stores the generated summaries in local JSONL caches. The same preprocessing module then creates two embedding views: one from the LLM-generated summaries and one from the original raw paper text, where raw text is formed by concatenating `title` and `abstract`.

The LLM model and embedding model are expected to be deployed with vLLM. The generated summary and raw-text embedding caches are consumed by the dual-path MMoE reranker during inference or training. Default cache and dataset paths are defined in `reviewer_training_utils.py` and can be overridden from the command line.

## Inference

Run inference with the released checkpoint:

```bash
python train_paper_set_dual_path_mmoe.py \
  --mode test \
  --checkpoint data/checkpoints/paper_set_dual_path_mmoe_mean/best_model.pth
```

Override data or cache locations when needed:

```bash
python train_paper_set_dual_path_mmoe.py \
  --mode test \
  --checkpoint data/checkpoints/paper_set_dual_path_mmoe_mean/best_model.pth \
  --test_df data/test_dataset_qwen_sampled.parquet \
  --test_author_papers data/cache/test_sampled_author_papers.jsonl \
  --test_paper_cache data/cache/test_only_paper_qwen_summary_embedding.jsonl \
  --test_wollm_cache data/cache/test_wollm_qwen_summary_embedding.jsonl
```

## Training

Train the dual-path MMoE reranker from generated embeddings:

```bash
python train_paper_set_dual_path_mmoe.py --mode train
```

By default, training uses staged optimization: task 0 first learns true-reviewer confidence, then tasks 1 and 2 optimize ranking among true, similar, and wrong candidates. Use `--no-mmoe_staged` to train all tasks jointly.
