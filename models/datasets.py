import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import random

class ReviewerTrainDataset(Dataset):
    def __init__(self, df_path: str, sample_ratio:float = 1.0, ):
        self.df = pd.read_parquet(df_path).reset_index(drop=True)
        self.nex_indexs = None
        if sample_ratio < 1:
            self.new_df = self.df.groupby('Qwen_Category_L1', group_keys=False).apply(lambda x: x.sample(frac=sample_ratio))
            self.nex_indexs = self.new_df.index.tolist()
            print(f"Dataset sampled with ratio {sample_ratio}, new size: {len(self.new_df)}")
        
        self.cache = {}
        if 'train' in df_path:
            with open('data/cache/train_summary_embedding.jsonl', 'r') as f:
                for line in tqdm(f, desc="Loading Train Embeddings"):
                    item = json.loads(line)
                    self.cache[item['id']] = np.array(item['embedding'])
        else: # val
            with open('./data/cache/val_summary_embedding.jsonl', 'r') as f:
                for line in tqdm(f, desc="Loading Val Embeddings"):
                    item = json.loads(line)
                    self.cache[item['id']] = np.array(item['embedding'])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.nex_indexs is not None:
            if idx not in self.nex_indexs:
                return None

        paper_id = row["Paper_OAG_ID"]
        reviewerIDs = row['Reviewer_OAG_IDs']
        qwen_category = row["Qwen_Category_L3"]
        random.seed(idx)
        wrong_candidate_ids = random.sample(list(row["Unqualified_Candidate_OAG_IDs"]), len(reviewerIDs))
        similar_candidate_ids = random.sample(list(row["Potential_Candidate_OAG_IDs"]), len(reviewerIDs))
        
        paper_embedding = torch.tensor(self.cache['paper:' + paper_id], dtype=torch.float32).unsqueeze(0)  # [1, emb_dim]

        reviewer_embeddings = []
        for reviewer_id in reviewerIDs:
            if 'reviewer:' + reviewer_id not in self.cache:
                raise ValueError(f"Embedding for reviewer {reviewer_id} not found in cache.")
            reviewer_embeddings.append(self.cache['reviewer:' + reviewer_id])
        wrong_embeddings = []
        for wrong_id in wrong_candidate_ids:
            if 'reviewer:' + wrong_id not in self.cache:
                raise ValueError(f"Embedding for reviewer {wrong_id} not found in cache.")
            wrong_embeddings.append(self.cache['reviewer:' + wrong_id])

        similar_embeddings = []
        for similar_id in similar_candidate_ids:
            if 'reviewer:' + similar_id not in self.cache:
                raise ValueError(f"Embedding for reviewer {similar_id} not found in cache.")
            similar_embeddings.append(self.cache['reviewer:' + similar_id])
        
        reviewer_embeddings = torch.tensor(np.array(reviewer_embeddings), dtype=torch.float32)  # [candidates, emb_dim]
        wrong_embeddings = torch.tensor(np.array(wrong_embeddings), dtype=torch.float32)
        similar_embeddings = torch.tensor(np.array(similar_embeddings), dtype=torch.float32)

        confidence_embeddings = torch.cat([reviewer_embeddings, wrong_embeddings], dim=0)  # [num_reviewers*2, emb_dim]
        recommendation_embeddings = torch.cat([reviewer_embeddings, similar_embeddings], dim=0)  # [num_reviewers*2, emb_dim]
        confidence_label = torch.tensor([1] * len(reviewerIDs) + [0] * len(wrong_candidate_ids), dtype=torch.float)  # [num_reviewers*2]
        recommendation_label = torch.tensor([1] * len(reviewerIDs) + [0] * len(similar_candidate_ids), dtype=torch.float)  # [num_reviewers*2]

        perm = torch.randperm(confidence_embeddings.size(0))
        confidence_embeddings = confidence_embeddings[perm]
        confidence_label = confidence_label[perm]
        perm = torch.randperm(recommendation_embeddings.size(0))
        recommendation_embeddings = recommendation_embeddings[perm]
        recommendation_label = recommendation_label[perm]

        return {
            "emb_paper": paper_embedding,
            "emb_conf": confidence_embeddings,
            "label_conf": confidence_label,
            "emb_recommendation": recommendation_embeddings,
            "label_recommendation": recommendation_label,
        }
    
class ReviewerTrainDataloader(DataLoader):
    def __init__(self, dataset: ReviewerTrainDataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        emb_paper = []
        emb_conf = []
        label_conf = []
        emb_recommendation = []
        label_recommendation = []
        for item in batch:
            emb_paper.append(item["emb_paper"].expand(item["emb_conf"].size(0), -1))
            emb_conf.append(item["emb_conf"])
            label_conf.append(item["label_conf"])
            emb_recommendation.append(item["emb_recommendation"])
            label_recommendation.append(item["label_recommendation"])
        return {
            "emb_paper": torch.cat(emb_paper, dim=0),
            "emb_conf": torch.cat(emb_conf, dim=0),
            "label_conf": torch.cat(label_conf, dim=0),
            "emb_recommendation": torch.cat(emb_recommendation, dim=0),
            "label_recommendation": torch.cat(label_recommendation, dim=0),
        }

class ReviewerTestDataset(Dataset):    
    def __init__(self, df_path: str = '/home/sunpenglei/hyh/candidate_tower/test_dataset_qwen_sampled.parquet'):
        self.df = pd.read_parquet(df_path)

        self.cache = {}
        self.difficulty = 0
        with open('./data/cache/test_summary_embedding.jsonl', 'r') as f:
            for line in f:
                item = json.loads(line)
                self.cache[item['id']] = np.array(item['embedding'])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        paper_id = row["Paper_OAG_ID"]
        reviewerIDs = row['Reviewer_OAG_IDs']
        wrong_candidate_ids = list(row["Unqualified_Candidate_OAG_IDs"])
        similar_candidate_ids = list(row["Potential_Candidate_OAG_IDs"])
        
        paper_embedding = torch.tensor(self.cache['paper:' + paper_id], dtype=torch.float32).unsqueeze(0)  # [1, emb_dim]

        reviewer_embeddings = []
        wrong_embeddings = []
        similar_embeddings = []
        for reviewer_id in reviewerIDs:
            if 'reviewer:' + reviewer_id not in self.cache:
                raise ValueError(f"Embedding for reviewer {reviewer_id} not found in cache.")
            reviewer_embeddings.append(self.cache['reviewer:' + reviewer_id])
        for wrong_id in wrong_candidate_ids:
            if 'reviewer:' + wrong_id not in self.cache:
                raise ValueError(f"Embedding for reviewer {wrong_id} not found in cache.")
            wrong_embeddings.append(self.cache['reviewer:' + wrong_id])
        for similar_id in similar_candidate_ids:
            if 'reviewer:' + similar_id not in self.cache:
                raise ValueError(f"Embedding for reviewer {similar_id} not found in cache.")
            similar_embeddings.append(self.cache['reviewer:' + similar_id])
        
        reviewer_embeddings = torch.tensor(np.array(reviewer_embeddings), dtype=torch.float32)  # [candidates, emb_dim]
        wrong_embeddings = torch.tensor(np.array(wrong_embeddings), dtype=torch.float32)
        similar_embeddings = torch.tensor(np.array(similar_embeddings), dtype=torch.float32)
        
        return {
            "emb_paper": paper_embedding,
            "emb_reviewer": reviewer_embeddings,
            "emb_wrong": wrong_embeddings,
            "emb_similar": similar_embeddings,
            "category": row["Qwen_Category_1"]
        }

class ReviewerTestDataloader(DataLoader):    
    def __init__(self, dataset: ReviewerTestDataset, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        emb_paper = []
        emb_reviewer = []
        emb_wrong = []
        emb_similar = []
        category = []
        for item in batch:
            emb_paper.append(item["emb_paper"])
            emb_reviewer.append(item["emb_reviewer"])
            emb_wrong.append(item["emb_wrong"])
            emb_similar.append(item["emb_similar"])
            category.append(item["category"])
        return {
            "emb_paper": torch.cat(emb_paper, dim=0),
            "emb_reviewer": torch.cat(emb_reviewer, dim=0),
            "emb_wrong": torch.cat(emb_wrong, dim=0),
            "emb_similar": torch.cat(emb_similar, dim=0),
            "category": category
        }