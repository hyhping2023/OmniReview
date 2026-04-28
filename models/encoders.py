import torch
import json, os
import threading
from typing import List, Union

def vllm_embedding_online(texts:list, vllm_url, dimensions=1024) -> torch.Tensor:
    import requests
    # print(texts)
    if vllm_url.endswith('8004'):
        model_name = "bge-m3"
    elif vllm_url.endswith('9002'):
        model_name = "qwen3-4b-emb"
    else:
        raise ValueError("不支持的vLLM服务地址")
    payload = {
        "model": model_name,  # 模型名称
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
    """编码器（支持冻结与微调）"""
    
    def __init__(self, dim=1024, model='qwen'):
        if model == 'qwen':
            self.vllm_url = "http://localhost:9002"
        elif model == 'bge-m3':
            self.vllm_url = "http://localhost:8004"
        else:
            raise ValueError("不支持的模型类型")
        self.dim = dim
        self.model = model
    
    def encode(self, ids, texts: Union[str, List[str]], output_type) -> torch.Tensor:
        """批量编码文本"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = vllm_embedding_online(texts, self.vllm_url, self.dim)
        if embeddings is None:
            raise ValueError("vLLM返回异常，未获取到embedding")
        with open(f'data/cache/{output_type}_{self.model}_summary_embedding.jsonl', 'a') as f:
            for id, emb in zip(ids, embeddings):
                f.write(json.dumps({
                    'id': id,
                    'embedding': emb.tolist()
                }) + '\n')
        return embeddings

if __name__ == "__main__":
    model = 'qwen'
    from tqdm import tqdm
    encoder = SummaryEncoder(model=model)
    cache = {}
    for output_type in ['test']:
        if os.path.exists(f'data/cache/{output_type}_{model}_summary_embedding.jsonl'):
            with open(f'data/cache/{output_type}_{model}_summary_embedding.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    cache[data['id']] = 1
            print(f"已有{len(cache)}条缓存")
        if os.path.exists(f'data/cache/{output_type}_only_paper_{model}_summary_embedding.jsonl'):
            with open(f'data/cache/{output_type}_only_paper_{model}_summary_embedding.jsonl', 'r') as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    cache[data['id']] = 1
            print(f"已有{len(cache)}条缓存")
        with open(f'data/cache/llm_only_paper_summary_{output_type}.jsonl', 'r') as f:
            ids = []
            texts = []
            for line in f:
                data = json.loads(line)
                if data['id'] in cache:
                    continue
                ids.append(data['id'])
                texts.append(data['summary'])
        print(f"需要编码{len(texts)}条数据")
        encoder.encode(ids, texts, output_type=f"{output_type}_only_paper")