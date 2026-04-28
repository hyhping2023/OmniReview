import requests
import json
from typing import Dict, List
from functools import lru_cache
from tqdm import tqdm
import os
import threading
import random

# ============================================================
# Global Port Configuration & Load Balancing
# ============================================================
LLM_PORTS = [20000,]
LLM_HOST = 'localhost'


def get_next_port() -> int:
    """Random port allocation for natural load balancing across processes"""
    return random.choice(LLM_PORTS)

def llm_query(messages: list, url: str = LLM_HOST, port: int = None) -> str:
    """调用LLM接口，支持端口负载均衡"""
    if port is None:
        port = get_next_port()
    
    vllm_url = f"http://{url}:{port}/v1/chat/completions"
    payload = {
            "model": "model",
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    response = requests.post(vllm_url, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    result = response.json()
    summary = result['choices'][0]['message']['content'].strip()
    return summary

# ============================================================
# Prompt Templates
# ============================================================
system_prompt_template = '''
You are an expert academic reviewer assistant. Generate a concise, critical summary of the research paper based ONLY on the title and abstract provided. Your summary must help peer reviewers quickly assess the paper's suitability for publication.

**Critical Requirements:**
1. **STRICT WORD LIMIT**: 150-180 words maximum. Every word must add value.
2. **SOURCE RESTRICTION**: Use ONLY the provided title and abstract. Never invent details, methods, results, or context not explicitly stated.
3. **REVIEWER-FOCUSED CONTENT**:
   - Core research question and its significance
   - Methodology assessment (rigor, appropriateness based on abstract description)
   - Key findings and their reliability
   - Novel contributions vs. existing literature
   - Major limitations explicitly mentioned
   - Overall publication recommendation likelihood

4. **TONE & STYLE**:
   - Objective, professional academic tone
   - Critical but constructive evaluation
   - Avoid promotional language or excessive praise
   - Use precise terminology from the field

5. **STRUCTURE**: Single coherent paragraph with logical flow: Problem → Methods → Findings → Contribution → Limitations → Assessment.

**WARNING**: If the abstract lacks crucial details (methods, limitations, contributions), explicitly state these gaps as review concerns. Never compensate with external knowledge.
'''

paper_summary_template = '''
Please provide a critical review summary for peer reviewers based on the following paper title and abstract. Strictly adhere to the word limit and guidelines.
Title: "{}"
Abstract: "{}"
'''

user_system_prompt_template = '''
You are an expert academic analyst. Your task is to analyze article summaries written by a candidate to identify their research domain expertise patterns. Focus exclusively on their demonstrated understanding of different research fields, key problems, and domain-specific significance.

**Core Analysis Dimensions:**
- **Domain Coverage**: Range of research fields covered across the summaries
- **Field-Specific Insight**: Depth of understanding demonstrated in each domain (terminology, key challenges, seminal works)
- **Problem Recognition**: Ability to identify domain-specific research problems and their importance
- **Contribution Evaluation**: Skill in assessing what constitutes meaningful contribution within specific fields
- **Field Context Awareness**: Understanding of how research fits within broader domain landscape
- **Terminology Mastery**: Appropriate use of domain-specific concepts and jargon
- **Cross-Domain Connections**: Recognition of relationships between different research areas

**Output Requirements:**
1. **Primary Research Domains** (bullet points): 2-3 fields where candidate shows strongest expertise
2. **Secondary Domains** (bullet points): Additional fields with demonstrated understanding  
3. **Domain Expertise Pattern** (concise paragraph): How they approach and evaluate different research fields
4. **Field-Specific Strengths** (2 bullet points): Most notable domain understanding capabilities

**Critical Guidelines:**
- Base analysis ONLY on observable domain knowledge from the summaries
- Focus on field-specific understanding rather than general analytical skills
- Prioritize research domain expertise over writing style or structural preferences
- Avoid inferring methodological assessment abilities from summary content
- Maintain descriptive neutrality - characterize domain expertise without judgment
- Word limit: 200 words maximum
'''

summaries_prompt = '''
Please analyze the following article summaries and provide a comprehensive characterization of their distinctive traits and working patterns. Focus on identifying observable characteristics rather than making suitability judgments.

**Candidate's Articles Summaries:**
{}
'''

class LLMSummarizer:
    def __init__(self, output_type, config: Dict = {"llm": {"cache_file": ""}}):
        self.cache_file = config["llm"]["cache_file"]
        self.output_type = output_type
        self.authorID2publications = self._load_oag_info()
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """加载LLM缓存"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return {json.loads(line)["id"]: json.loads(line) for line in f}
        return {}
    
    def _build_paper_summary_messages(self, title:str, abstract:str) -> list:
        """应用Prompt模板"""
        return [
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": paper_summary_template.format(title, abstract)}
        ]
    
    def _build_reviewer_summary_messages(self, summaries: str) -> list:
        """应用Reviewer Prompt模板"""
        return [
            {"role": "system", "content": user_system_prompt_template},
            {"role": "user", "content": summaries_prompt.format(summaries)}
        ]
    
    def _load_oag_info(self):
        authorid2publications = {}
        print("😋 Loading author publications...")
        with open(f'{self.output_type}_sampled_author_papers.jsonl', 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                authorid2publications[data['author_id']] = data['publications']
        print("😋 Loaded author publications!")
        print(f"Total authors loaded: {len(authorid2publications)}")
        return authorid2publications
    
    def summarize_paper(self, paper_id: str, title: str, abstract: str) -> str:
        """生成论文摘要（带缓存）"""
        cache_key = f"paper:{paper_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]["summary"]
        
        messages = self._build_paper_summary_messages(title, abstract)
        
        try:
            summary = llm_query(messages)
            
            # 写入缓存
            with open(self.cache_file, "a") as f:
                f.write(json.dumps({"id": cache_key, "summary": summary}) + "\n")
            self.cache[cache_key] = {"summary": summary}
            return summary
        except Exception as e:
            print(f"LLM调用失败 {paper_id}: {e}")
            # 返回默认摘要
            return ''
    
    def summarize_reviewer(self, reviewer_id: str, topk = 10, citation_topk=5) -> str:
        """根据审稿人所有论文的摘要，总结其研究专长（带缓存）"""
        cache_key = f"reviewer:{reviewer_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]["expertise_summary"]
        
        reviewer_papers_summaries = ""
        publications = self.authorID2publications.get(reviewer_id, [])
        if len(publications) == 0:
            print(f"⚠️ 审稿人 {reviewer_id} 无论文数据")
        if len(publications) <= 10:
            filtered_pubs = publications
        else:
            n_citation_publications = [pub for pub in publications if pub.get('n_citation') > 0]
            n_citation_publications.sort(key=lambda x: x['n_citation'], reverse=True)
            if len(n_citation_publications) >= citation_topk:
                filtered_pubs = n_citation_publications[:citation_topk]
                left = topk - citation_topk
            else:
                filtered_pubs = n_citation_publications
                left = topk - len(n_citation_publications)
            if left > 0:
                remaining_pubs = [pub for pub in publications if pub not in filtered_pubs]
                remaining_pubs.sort(key=lambda x: x.get('year', 0), reverse=True)
                filtered_pubs += remaining_pubs[:left]
        for idx, pub in enumerate(filtered_pubs):
            title = pub.get('title', '')
            abstract = pub.get('abstract', '')
            if title and abstract:
                paper_summary = self.summarize_paper(pub['id'], title, abstract)
                reviewer_papers_summaries += f"Summary{idx}:\n{paper_summary}\n\n"
        messages = self._build_reviewer_summary_messages(reviewer_papers_summaries)
        try:
            expertise_summary = llm_query(messages)
            with open(self.cache_file, "a") as f:
                f.write(json.dumps({"id": cache_key, "expertise_summary": expertise_summary}) + "\n")
            self.cache[cache_key] = {"expertise_summary": expertise_summary}
            return expertise_summary
        except Exception as e:
            print(f"LLM调用失败 {reviewer_id}: {e}")
            return ''
        
def summarize_paper(paper_id: str, messages: list, output_type: str, port: int = None) -> str:
    """生成论文摘要（带缓存，支持端口负载均衡）"""
    cache_key = f"paper:{paper_id}"
    cache_file = f"./data/cache/llm_cache_{output_type}.jsonl"

    try:
        summary = llm_query(messages, port=port)

        # 写入缓存
        with open(cache_file, "a") as f:
            f.write(json.dumps({"id": cache_key, "summary": summary}) + "\n")
        return summary
    except Exception as e:
        print(f"LLM调用失败 {paper_id}: {e}")
        return ''

def summarize_reviewer(reviewer_id: str, messages: list, output_type: str, port: int = None) -> str:
    """根据审稿人所有论文的摘要，总结其研究专长（带缓存，支持端口负载均衡）"""
    cache_key = f"reviewer:{reviewer_id}"
    cache_file = f"./data/cache/llm_cache_{output_type}.jsonl"

    try:
        expertise_summary = llm_query(messages, port=port)
        with open(cache_file, "a") as f:
            f.write(json.dumps({"id": cache_key, "summary": expertise_summary}) + "\n")
        return expertise_summary
    except Exception as e:
        print(f"LLM调用失败 {reviewer_id}: {e}")
        return ''

        
if __name__ == "__main__":
    # 测试LLM模块
    config = {"llm": {"cache_file": "./test_cache.jsonl"}}
    summarizer = LLMSummarizer('test', config)
    result = summarizer.summarize_paper("test123", "Test", "This is a test abstract.")
    print(result)