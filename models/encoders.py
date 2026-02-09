import torch
import json, os
from typing import List, Union

def vllm_embedding_online(texts:list, vllm_url, dimensions=1024) -> torch.Tensor:
    import requests
    # print(texts)
    payload = {
        "model": "qwen3-4b-emb",  # 模型名称
        "input": texts,
        "encoding_format": "float"
    }
    res = requests.post(f"{vllm_url}/v1/embeddings", json=payload)
    res_json = res.json()
    try:
        outputs = res_json['data']
        embeddings = torch.tensor([output['embedding'] for output in outputs])
        embeddings = embeddings[:, :dimensions]
        return embeddings
    except:
        return None

class SummaryEncoder:
    """编码器"""
    
    def __init__(self, dim=1024):
        self.vllm_url = "http://localhost:8003"
        self.dim = dim
    
    def encode(self, ids, texts: Union[str, List[str]], output_type) -> torch.Tensor:
        """批量编码文本"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = vllm_embedding_online(texts, self.vllm_url, self.dim)
        if embeddings is None:
            raise ValueError("vLLM返回异常，未获取到embedding")
        with open(f'./data/cache/{output_type}_summary_embedding.jsonl', 'a') as f:
            for id, emb in zip(ids, embeddings):
                f.write(json.dumps({
                    'id': id,
                    'embedding': emb.tolist()
                }) + '\n')
        return embeddings