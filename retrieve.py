"""
retrieve.py
统一封装检索流程：FAISS + BM25 + 语义 rerank
"""

from collections import Counter
import math
import re
from typing import Dict, List

import numpy as np


def tokenize_mixed(text: str) -> List[str]:
    """
    轻量中英混合分词：英文按单词，中文按单字。
    不依赖额外分词库，便于快速接入 BM25。
    """
    text = (text or "").lower()
    parts = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    return [p for p in parts if p.strip()]


class BM25Retriever:
    """简化 BM25 实现，用于关键词召回。"""

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize_mixed(doc) for doc in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

        self.term_df: Dict[str, int] = {}
        self.doc_tf: List[Counter] = []

        for toks in self.doc_tokens:
            tf = Counter(toks)
            self.doc_tf.append(tf)
            for term in tf.keys():
                self.term_df[term] = self.term_df.get(term, 0) + 1

        self.n_docs = len(docs)

    def _idf(self, term: str) -> float:
        df = self.term_df.get(term, 0)
        # BM25 常见平滑写法，避免负值过大
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_query(self, query: str) -> np.ndarray:
        q_terms = tokenize_mixed(query)
        if not q_terms:
            return np.zeros(self.n_docs, dtype=np.float32)

        scores = np.zeros(self.n_docs, dtype=np.float32)
        for i, tf in enumerate(self.doc_tf):
            dl = self.doc_lens[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))

            total = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                idf = self._idf(term)
                total += idf * (f * (self.k1 + 1)) / (f + denom_norm)
            scores[i] = float(total)

        return scores


def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = np.asarray(list(scores.values()), dtype=np.float32)
    min_v = float(vals.min())
    max_v = float(vals.max())
    if abs(max_v - min_v) < 1e-9:
        return {k: 1.0 for k in scores.keys()}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


def _to_cosine_sim(vectors: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    # 归一化后点积即余弦相似度
    q = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    x = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    return (x @ q.T).reshape(-1)


def hybrid_retrieve(
    query: str,
    model,
    index,
    chunk_texts: List[str],
    chunk_sources: List[int],
    bm25: BM25Retriever,
    top_k: int,
    faiss_candidate_k: int,
    bm25_candidate_k: int,
    hybrid_candidate_k: int,
    rerank_top_k: int,
    faiss_weight: float,
    bm25_weight: float,
):
    """
    混合检索主流程：
    1) FAISS 召回候选
    2) BM25 召回候选
    3) 分数归一化融合
    4) 语义 rerank
    """

    if not chunk_texts:
        return []

    query_vec = np.asarray(model.encode([query]), dtype=np.float32)

    # ===== 1) FAISS 候选 =====
    kf = min(max(1, faiss_candidate_k), len(chunk_texts))
    distances, indices = index.search(query_vec, k=kf)

    faiss_scores: Dict[int, float] = {}
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        # L2 距离转为相似度分值，数值越大越好
        faiss_scores[int(idx)] = 1.0 / (1.0 + float(dist))

    # ===== 2) BM25 候选 =====
    all_bm25_scores = bm25.score_query(query)
    kb = min(max(1, bm25_candidate_k), len(chunk_texts))
    bm25_top_indices = np.argsort(all_bm25_scores)[::-1][:kb]

    bm25_scores: Dict[int, float] = {
        int(i): float(all_bm25_scores[i]) for i in bm25_top_indices
    }

    # ===== 3) 融合打分 =====
    faiss_norm = _normalize(faiss_scores)
    bm25_norm = _normalize(bm25_scores)

    candidate_ids = set(faiss_norm.keys()) | set(bm25_norm.keys())
    hybrid_scores: Dict[int, float] = {}
    for cid in candidate_ids:
        f_score = faiss_norm.get(cid, 0.0)
        b_score = bm25_norm.get(cid, 0.0)
        hybrid_scores[cid] = faiss_weight * f_score + bm25_weight * b_score

    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_ids = [cid for cid, _ in sorted_hybrid[: max(1, hybrid_candidate_k)]]

    # ===== 4) rerank（语义相似度） =====
    candidate_texts = [chunk_texts[cid] for cid in candidate_ids]
    candidate_vecs = np.asarray(model.encode(candidate_texts), dtype=np.float32)
    rerank_scores = _to_cosine_sim(candidate_vecs, query_vec)

    reranked = []
    for local_i, cid in enumerate(candidate_ids):
        reranked.append(
            {
                "idx": int(cid),
                "source_id": int(chunk_sources[cid]),
                "text": chunk_texts[cid],
                "hybrid_score": float(hybrid_scores.get(cid, 0.0)),
                "rerank_score": float(rerank_scores[local_i]),
            }
        )

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    final_k = min(top_k, rerank_top_k, len(reranked))
    return reranked[:final_k]


def split_query(query: str) -> List[str]:
    """
    拆分用户输入的多个问题。
    按中文问号（？）和英文问号（?）切割，保留非空问题。
    
    示例：
    - "whyLang是什么？YOLO又是什么？" -> ["whyLang是什么", "YOLO又是什么"]
    - "单个问题" -> ["单个问题"]
    """
    parts = re.split(r'[，,。.！!；;？?]', query)
    results = [p.strip() for p in parts if p.strip()]
    return results if results else [query]


def hybrid_retrieve_with_query_split(
    query: str,
    model,
    index,
    chunk_texts: List[str],
    chunk_sources: List[int],
    bm25: BM25Retriever,
    top_k: int,
    faiss_candidate_k: int,
    bm25_candidate_k: int,
    hybrid_candidate_k: int,
    rerank_top_k: int,
    faiss_weight: float,
    bm25_weight: float,
):
    """
    支持 Query 拆分的混合检索：
    1) 按"？"和"?"拆分用户输入
    2) 对每个子问题分别检索
    3) 合并所有结果、去重、按 rerank_score 重排序
    """
    
    sub_queries = split_query(query)
    
    if not sub_queries:
        return []
    
    # 收集所有子查询的结果（按 idx 去重）
    all_results_dict: Dict[int, Dict] = {}
    
    for sub_query in sub_queries:
        sub_results = hybrid_retrieve(
            query=sub_query,
            model=model,
            index=index,
            chunk_texts=chunk_texts,
            chunk_sources=chunk_sources,
            bm25=bm25,
            top_k=top_k,
            faiss_candidate_k=faiss_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            hybrid_candidate_k=hybrid_candidate_k,
            rerank_top_k=rerank_top_k,
            faiss_weight=faiss_weight,
            bm25_weight=bm25_weight,
        )
        
        # 合并：如果已有该 idx，保留最高分
        for item in sub_results:
            idx = item["idx"]
            if idx not in all_results_dict:
                all_results_dict[idx] = item
            else:
                # 保留 rerank_score 更高的
                if item["rerank_score"] > all_results_dict[idx]["rerank_score"]:
                    all_results_dict[idx] = item
    
    # 所有结果按 rerank_score 降序排列
    merged_results = sorted(
        all_results_dict.values(),
        key=lambda x: x["rerank_score"],
        reverse=True
    )
    
    # 返回 n * RERANK_TOP_K 个结果（n 为子问题数量）
    # 确保每个子问题都有足够的信息覆盖，避免某些问题因分数不足而被淘汰
    num_sub_queries = len(sub_queries)
    max_results = num_sub_queries * rerank_top_k
    final_k = min(max_results, len(merged_results))
    return merged_results[:final_k]
