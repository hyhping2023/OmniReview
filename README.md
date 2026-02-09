# OmniReview

This is the official code repository for the paper **"OmniReview: A Large-scale Benchmark and LLM-enhanced Framework for Realistic Reviewer Recommendation"**.

## Overview

OmniReview is a large-scale benchmark and framework designed to address the challenges of realistic reviewer recommendation. It leverages advanced machine learning techniques, including Multi-gate Mixture-of-Experts (MMoE) models, and incorporates large language models (LLMs) to enhance the recommendation process.

The repository includes:

- **models/datasets**: Preprocessed datasets for training, validation, and testing.
- **models/encoders**: Implementation of embedding the summaries.
- **models/mmoe**: Implementation of the MMoE model for multi-task learning.
- **llm/summarizer**: Implementation of the LLM Summarizer.
- **data/category.json**: The Discipline Taxonomy.
- **preprocess**: Asynchronous usage of the LLM Summarizer.
- **main**: Scripts for training and validating the model and tools for evaluating the recommendation performance using metrics like NDCG and MAP.
- **disambiguation**: The script used for data sources disambiguation.

---

## Features

- **Multi-task Learning**: Supports confidence prediction and recommendation ranking tasks.
- **LLM Integration**: Utilizes large language models for summarizing papers and reviewers.
- **Scalable Framework**: Designed to handle large-scale datasets efficiently.
- **Custom Metrics**: Includes evaluation metrics such as NDCG, MAP, and Recall.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.4.0
- CUDA (if using GPU)
- pypinyin, numpy, pandas

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/hyhping2023/OmniReview.git
   cd OmniReview

   ```
2. Download the dataset from HuggingFace
    [https://huggingface.co/datasets/HYHPING2023/OmniReview](https://huggingface.co/datasets/HYHPING2023/OmniReview)
    
3. Download the LLM Model
    [https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
    [https://huggingface.co/Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)

4. Use VLLM to deploy the models.
