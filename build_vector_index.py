"""
build_vector_index.py
离线构建索引：读取知识库 → 分块 → 向量化 → 保存索引

使用方法：
    python build_vector_index.py
"""

import os
import pickle
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    DOCS_FILE,
    INDEX_DIR,
    INDEX_FILE,
    CHUNKS_FILE,
    VECTOR_INDEX_MODEL_NAME,
    MAX_LENGTH,
    OVERLAP,
)

# 调用以下模块：
from chunker import chunk_text # 文本分块模块


def build_index():
    """
    构建并保存 FAISS 索引和块信息
    
    流程：
    1. 读取知识库文档
    2. 分块处理
    3. 加载向量模型
    4. 编码为向量
    5. 构建 FAISS 索引
    6. 持久化保存
    """
    
    # ===== 步骤 1：读取文档 =====
    print(f"读取文档文件: {DOCS_FILE}")
    if not os.path.exists(DOCS_FILE):
        print(f"error: 找不到文件 {DOCS_FILE}")
        return False
    
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    
    if not docs:
        print(f"error: {DOCS_FILE} 为空")
        return False
    
    print(f"成功加载 {len(docs)} 段文档\n")
    
    # ===== 步骤 2：分块 =====
    print(f"进行文本分块 (MAX_LENGTH={MAX_LENGTH}, OVERLAP={OVERLAP})")
    chunk_texts = []
    chunk_sources = []
    
    for doc_id, doc in enumerate(docs):
        chunks = chunk_text(doc, max_length=MAX_LENGTH, overlap=OVERLAP)
        for chunk in chunks:
            chunk_texts.append(chunk)
            chunk_sources.append(doc_id)
    
    print(f"分块完成：共生成 {len(chunk_texts)} 个块\n")
    
    # ===== 步骤 3：加载向量模型 =====
    print(f"加载向量模型: {VECTOR_INDEX_MODEL_NAME}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"设备: {device}")
        model = SentenceTransformer(VECTOR_INDEX_MODEL_NAME, device=device)
        print(f"模型加载成功\n")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    # ===== 步骤 4：编码为向量 =====
    print(f"向量化 {len(chunk_texts)} 个块...")
    try:
        embeddings = model.encode(chunk_texts, show_progress_bar=True)
        vectors = np.asarray(embeddings, dtype=np.float32)
        print(f"向量化完成，维度: {vectors.shape}\n")
    except Exception as e:
        print(f"向量化失败: {e}")
        return False
    
    # ===== 步骤 5：构建 FAISS 索引 =====
    print(f"构建 FAISS 索引...")
    try:
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        print(f"FAISS 索引构建成功\n")
    except Exception as e:
        print(f"索引构建失败: {e}")
        return False
    
    # ===== 步骤 6：创建存储目录 =====
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # ===== 步骤 7：保存索引 =====
    print(f"保存索引文件...")
    try:
        faiss.write_index(index, INDEX_FILE)
        print(f"FAISS 索引: {INDEX_FILE}")
    except Exception as e:
        print(f"保存 FAISS 索引失败: {e}")
        return False
    
    try:
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump({"texts": chunk_texts, "sources": chunk_sources}, f)
        print(f"块信息: {CHUNKS_FILE}")
        print(f"所有文件保存成功\n")
    except Exception as e:
        print(f"保存块信息失败: {e}")
        return False
    
    # ===== 完成 =====
    print("=" * 50)
    print("索引构建完成！")
    print("=" * 50)
    print(f"文档数: {len(docs)}")
    print(f"块数: {len(chunk_texts)}")
    print(f"向量维度: {dim}")
    print(f"索引文件: {INDEX_FILE}")
    print(f"块信息文件: {CHUNKS_FILE}")
    print("\n现在可以运行 rag_demo.py 进行检索了")
    
    return True


if __name__ == "__main__":
    success = build_index()
    exit(0 if success else 1)
