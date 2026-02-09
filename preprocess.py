import pandas as pd
from tqdm import tqdm
import os  # 引入 os 模块用于文件判断
import json
import random
import pandas as pd
import multiprocessing as mp
from llm.summarizer import LLMSummarizer, summarize_paper, summarize_reviewer
from models.encoders import SummaryEncoder
    
def author_id_extract(target='train'):
    file = f'path/to/{target}_dataset_qwen_sampled.parquet'
    df = pd.read_parquet(file)
    author_ids = set()
    for idx, row in enumerate(df.itertuples()):
        reviewer_ids = row.Reviewer_IDs.tolist()

        # Train/Val New
        if target == 'train' or target == 'val':
            random.seed(idx)
            wrong_candidate_ids = random.sample(row.Wrong_Candidates.tolist(), len(reviewer_ids))
            similar_candidate_ids = random.sample(row.Similar_Candidates.tolist(), len(reviewer_ids))
            ids = set(reviewer_ids + wrong_candidate_ids + similar_candidate_ids)

        else:
            wrong_candidates_ids = row.Wrong_Candidates.tolist()
            similar_candidates_ids = row.Similar_Candidates.tolist()
            ids = set(reviewer_ids + wrong_candidates_ids + similar_candidates_ids)
        print(f"Row {idx}: Collected {len(ids)} unique publication IDs from reviewers and candidates.")
        author_ids.update(ids)
    print(f"Total unique publication IDs collected: {len(author_ids)}")
    with open(f'data/cache/{target}_sampled_authorIDs.txt', 'w') as f:
        for aid in author_ids:
            f.write(f"{aid}\n")

def paper_pre_summarize_new(target='train'):
    pool = mp.Pool(processes=64)
    papers = set()
    output_type = target
    llm = LLMSummarizer(output_type=output_type)

    if os.path.exists(f'./data/cache/llm_cache_{output_type}.jsonl'):
        with open(f'./data/cache/llm_cache_{output_type}.jsonl', 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                if data['id'].startswith('paper:'):
                    paper_id = data['id'][6:]
                    papers.add(paper_id)
        print(f"Already cached papers: {len(papers)}")

    with open(f'path/to/{output_type}_dataset_qwen_sampled.parquet', 'rb') as f:
        df = pd.read_parquet(f)
        for idx, row in enumerate(tqdm(df.itertuples(), total=len(df))):
            paper_id = row.ID
            paper_title = row.Title
            paper_abstract = row.Abstract

            reviewer_ids = row.Reviewer_IDs.tolist()
            if output_type == 'train' or output_type == 'val':
                random.seed(idx)
                wrong_candidate_ids = random.sample(row.Wrong_Candidates.tolist(), len(reviewer_ids))
                similar_candidate_ids = random.sample(row.Similar_Candidates.tolist(), len(reviewer_ids))
            else:
                wrong_candidate_ids = row.Wrong_Candidates.tolist()
                similar_candidate_ids = row.Similar_Candidates.tolist()

            if paper_id not in papers:
                pool.apply_async(summarize_paper, args=(paper_id, llm._build_paper_summary_messages(paper_title, paper_abstract)))
                papers.add(paper_id)

            for reviewer_id in reviewer_ids:
                for paper in get_candidate_papers(llm, reviewer_id, topk=10, citation_topk=5):
                    if paper['id'] not in papers:
                        pool.apply_async(summarize_paper, args=(paper['id'], llm._build_paper_summary_messages(paper['title'], paper['abstract']), output_type))
                        papers.add(paper['id'])
            for wrong_id in wrong_candidate_ids:
                for paper in get_candidate_papers(llm, wrong_id, topk=10, citation_topk=5):
                    if paper['id'] not in papers:
                        pool.apply_async(summarize_paper, args=(paper['id'], llm._build_paper_summary_messages(paper['title'], paper['abstract']), output_type))
                        papers.add(paper['id'])
            for similar_id in similar_candidate_ids:
                for paper in get_candidate_papers(llm, similar_id, topk=10, citation_topk=5):
                    if paper['id'] not in papers:
                        pool.apply_async(summarize_paper, args=(paper['id'], llm._build_paper_summary_messages(paper['title'], paper['abstract']), output_type))
                        papers.add(paper['id'])
    print(f"Total unique papers to summarize: {len(papers)}")
    pool.close()
    pool.join()

def reviewer_pre_summarize_new(target='train'):
    pool = mp.Pool(processes=64)
    reviewers = set()
    paper_pre_summaries = {}
    output_type = target
    llm = LLMSummarizer(output_type=output_type)

    with open(f'./data/cache/llm_cache_{output_type}.jsonl', 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            if data['id'].startswith('paper:'):
                paper_id = data['id'][6:]
                paper_pre_summaries[paper_id] = data['summary']
            if data['id'].startswith('reviewer:'):
                reviewer_id = data['id'][9:]
                reviewers.add(reviewer_id)
    print(f"Already cached paper summaries: {len(paper_pre_summaries)}")
    print(f"Already cached reviewers: {len(reviewers)}")

    with open(f'path/to/{output_type}_dataset_qwen_sampled.parquet', 'rb') as f:
        df = pd.read_parquet(f)
        for idx, row in enumerate(tqdm(df.itertuples(), total=len(df))):
            reviewer_ids = row.Reviewer_IDs.tolist()
            if output_type == 'train' or output_type == 'val':
                random.seed(idx)
                wrong_candidate_ids = random.sample(row.Wrong_Candidates.tolist(), len(reviewer_ids))
                similar_candidate_ids = random.sample(row.Similar_Candidates.tolist(), len(reviewer_ids))
            else:
                wrong_candidate_ids = row.Wrong_Candidates.tolist()
                similar_candidate_ids = row.Similar_Candidates.tolist()

            for reviewer_id in reviewer_ids:
                reviewer_papers_summaries = ""
                if reviewer_id not in reviewers:
                    for paper in get_candidate_papers(llm, reviewer_id, topk=10, citation_topk=5):
                        paper_summary = paper_pre_summaries[paper['id']]
                        reviewer_papers_summaries += f"Summary{idx}:\n{paper_summary}\n\n"
                    messages = llm._build_reviewer_summary_messages(reviewer_papers_summaries)
                    pool.apply_async(summarize_reviewer, args=(reviewer_id, messages, output_type))
                    reviewers.add(reviewer_id)
            for wrong_id in wrong_candidate_ids:
                if wrong_id not in reviewers:
                    for paper in get_candidate_papers(llm, wrong_id, topk=10, citation_topk=5):
                        paper_summary = paper_pre_summaries[paper['id']]
                        reviewer_papers_summaries += f"Summary{idx}:\n{paper_summary}\n\n"
                    messages = llm._build_reviewer_summary_messages(reviewer_papers_summaries)
                    pool.apply_async(summarize_reviewer, args=(wrong_id, messages, output_type))
                    reviewers.add(wrong_id)
            for similar_id in similar_candidate_ids:
                if similar_id not in reviewers:
                    for paper in get_candidate_papers(llm, similar_id, topk=10, citation_topk=5):
                        paper_summary = paper_pre_summaries[paper['id']]
                        reviewer_papers_summaries += f"Summary{idx}:\n{paper_summary}\n\n"
                    messages = llm._build_reviewer_summary_messages(reviewer_papers_summaries)
                    pool.apply_async(summarize_reviewer, args=(similar_id, messages, output_type))
                    reviewers.add(similar_id)

    print(f"Total unique reviewers to summarize: {len(reviewers)}")
    pool.close()
    pool.join()

def summary_embedding(target='train'):
    encoder = SummaryEncoder()
    cache = {}
    output_type = target # train/val/test
    if os.path.exists(f'./data/cache/{output_type}_summary_embedding.jsonl'):
        with open(f'./data/cache/{output_type}_summary_embedding.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                cache[data['id']] = 1
        print(f"Already {len(cache)} caches found for {output_type} summaries.")
    with open(f'data/cache/{output_type}_final_summaries.jsonl', 'r') as f:
        ids = []
        texts = []
        for line in f:
            data = json.loads(line)
            if data['id'] in cache:
                continue
            ids.append(data['id'])
            texts.append(data['summary'])
    print(f"Need to embed {len(texts)} summaries for {output_type} set.")
    encoder.encode(ids, texts, output_type=output_type)

if __name__ == "__main__":
    os.makedirs('data/cache', exist_ok=True)
    author_id_extract()
    # paper_pre_embedding_new()
    # reviewer_pre_embedding_new()
    # summary_embedding()