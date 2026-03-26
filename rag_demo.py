# 这是一个RAG示例，展示了如何从已编码的索引进行向量检索

import os
import pickle
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 导入配置和功能模块
from config import (
    INDEX_FILE,
    CHUNKS_FILE,
	VECTOR_INDEX_MODEL_NAME, # 向量编码模型名称
    TOP_K, # 检索返回的块数量
    MAX_COUNT, # 最多允许的查询次数，防止无限循环
	LLM_MODEL_NAME, # 使用的LLM大模型名称
	SHOW_BEST_ONLY, # 是否仅显示最佳结果
	FAISS_CANDIDATE_K,
	BM25_CANDIDATE_K,
	HYBRID_CANDIDATE_K,
	RERANK_TOP_K,
	FAISS_SCORE_WEIGHT,
	BM25_SCORE_WEIGHT,
	LLM_PROVIDER,
	QWEN_MODEL_NAME,
)

# 调用以下模块：
from deepseek_API import generate_with_deepseek # LLM生成接口
from Qwen_API import generate_with_qwen, preload_qwen_model # 本地Qwen生成接口
from retrieve import BM25Retriever, hybrid_retrieve_with_query_split # 检索模块


def load_index_and_chunks():
	"""
	加载已保存的 FAISS 索引和块信息
	
	Returns:
		tuple: (index, chunk_texts, chunk_sources) 或 (None, None, None) 失败时
	"""
	# 检查索引文件是否存在
	if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
		print(f"   error:缺少索引文件")
		print(f"   请先运行: python build_vector_index.py")
		print(f"   缺失文件:")
		if not os.path.exists(INDEX_FILE):
			print(f"     - {INDEX_FILE}")
		if not os.path.exists(CHUNKS_FILE):
			print(f"     - {CHUNKS_FILE}")
		return None, None, None
	
	# 加载 FAISS 索引
	try:
		index = faiss.read_index(INDEX_FILE)
		print(f"加载 FAISS 索引: {INDEX_FILE}")
	except Exception as e:
		print(f"加载索引失败: {e}")
		return None, None, None
	
	# 加载块信息
	try:
		with open(CHUNKS_FILE, "rb") as f:
			data = pickle.load(f)
			chunk_texts = data["texts"]
			chunk_sources = data["sources"]
		print(f"加载块信息: {CHUNKS_FILE}")
		print(f"块数: {len(chunk_texts)}")
	except Exception as e:
		print(f"加载块信息失败: {e}")
		return None, None, None
	
	return index, chunk_texts, chunk_sources


def main():
	# 1) 加载向量模型
	print("加载向量模型...")
	try:
		# 自动选择设备：优先 GPU，降级到 CPU
		device = "cuda" if torch.cuda.is_available() else "cpu"
		print(f"设备: {device}")
		model = SentenceTransformer(VECTOR_INDEX_MODEL_NAME, device=device)
		print(f"模型加载成功\n")
	except Exception as e:
		print(f"模型加载失败: {e}")
		return
	
	# 2) 加载已保存的索引
	print("加载已保存的索引...")
	index, chunk_texts, chunk_sources = load_index_and_chunks()
	if index is None:
		return
	print()

	# Qwen 模型预加载（如果配置使用 Qwen）
	if LLM_PROVIDER == "qwen_local":
		print("预加载本地 Qwen 模型...")
		preload_ok = preload_qwen_model()
		if not preload_ok:
			print("警告: Qwen 预加载失败，首次问答可能会变慢。")
		print()
	
	# ====== 3) 交互式查询 ======

	print("RAG 检索系统已就绪")
	print("检索模式: FAISS + BM25 + Rerank")
	print(f"当前大模型提供方: {LLM_PROVIDER}")
	bm25 = BM25Retriever(chunk_texts)

	if SHOW_BEST_ONLY:
		print("当前配置: 仅显示模型结果 (SHOW_BEST_ONLY=True)")
	else:		
		print("当前配置: 显示完整检索结果 (SHOW_BEST_ONLY=False)")
	
	print("\n" + "-" * 100 + "\n")# 分隔符，清晰区分每次查询

	counter = 0
	while counter < MAX_COUNT:  # 最多允许 MAX_COUNT 次查询，防止无限循环

		# ====== 用户输入 ======
		print( "=" * 50)
		print(f"请输入文本 (直接回车退出, 查询第 {counter + 1}/{MAX_COUNT} 次):")
		print("=" * 50)
		query = input("> ").strip()
		if not query:
			print("程序结束。")
			break

		counter += 1

		# 执行混合检索 + rerank（支持 Query 拆分）
		try:
			results = hybrid_retrieve_with_query_split(
				query=query,
				model=model,
				index=index,
				chunk_texts=chunk_texts,
				chunk_sources=chunk_sources,
				bm25=bm25,
				top_k=TOP_K,
				faiss_candidate_k=FAISS_CANDIDATE_K,
				bm25_candidate_k=BM25_CANDIDATE_K,
				hybrid_candidate_k=HYBRID_CANDIDATE_K,
				rerank_top_k=RERANK_TOP_K,
				faiss_weight=FAISS_SCORE_WEIGHT,
				bm25_weight=BM25_SCORE_WEIGHT,
			)
		except Exception as e:
			print(f"检索失败: {e}")
			continue

		if not results:
			print("未检索到可用结果。")
			continue
		
		k = len(results)
		top_contexts = [item["text"] for item in results]
		
		if not SHOW_BEST_ONLY:  # 
			# ========== 输出最佳结果 =========
			best_idx = results[0]["idx"]
			# best_dist = distances[0][0]
			# best_source = chunk_sources[best_idx]
			print("\n" + "=" * 50)
			print("最合适的结果")
			print("=" * 50)
			print(f"{chunk_texts[best_idx]}")
		
			# ======== 输出完整检索结果 ========
			print("\n" + "=" * 50)
			print("完整检索结果")
			print("=" * 50)
			print(f"问题: {query}")
			print(f"最相关分块 (Top-{k}):")
			for rank, item in enumerate(results, start=1):
				source_id = item["source_id"]
				content = item["text"]
				content_preview = content[:60] + "..." if len(content) > 60 else content
				print(
					f"{rank}. [rerank={item['rerank_score']:.4f}, hybrid={item['hybrid_score']:.4f}] "
					f"文档#{source_id} | {content_preview}"
				)

		# ======== 按配置调用 LLM 并输出回答 ========
		if LLM_PROVIDER == "deepseek":
			answer = generate_with_deepseek(query, top_contexts)
			answer_model_name = LLM_MODEL_NAME
		elif LLM_PROVIDER == "qwen_local":
			answer = generate_with_qwen(query, top_contexts)
			answer_model_name = QWEN_MODEL_NAME
		else:
			print(f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}")
			print("请在 config.py 中将 LLM_PROVIDER 设置为 deepseek 或 qwen_local")
			answer = None
			answer_model_name = "unknown"

		if answer:
			print("\n" + "=" * 50)
			print(f"回答来自: {answer_model_name}")
			print("=" * 50)
			print(f"> {answer}")
		
		print("\n" + "-" * 100 + "\n")# 分隔符，清晰区分每次查询
		# 继续下一轮查询


if __name__ == "__main__":
	main()
