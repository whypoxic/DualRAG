# RAG 系统 - 运行说明

本文档介绍本项目的运行方式与关键配置。

---

## 1. 项目文件结构

```
RAG/
├── config.py              # 全局配置文件
├── chunker.py             # 文本分块模块
├── build_vector_index.py  # 索引构建脚本
├── deepseek_API.py        # DeepSeek 调用接口
├── Qwen_API.py            # 本地 Qwen 调用接口
├── prompt_template.py     # LLM 共用 Prompt 模板
├── retrieve.py            # 混合检索模块（采用FAISS + BM25 + rerank）
├── rag_demo.py            # RAG 主程序
├── download_bge_model.py  # 向量模型下载脚本
├── download_Qwen_model.py # Qwen 模型下载脚本
├── docs.txt               # 知识库文档
├── indexes/               # 索引存储目录
│   ├── vectors.faiss      # FAISS 向量索引
│   └── chunks.pkl         # 块文本和源信息
├── bge-small-zh/          # 本地向量模型目录
├── qwen_local/            # 本地 Qwen 模型目录
└── RAG运行说明.md         # 本文档
```

---

## 2. 工作流程

### 首次使用

#### 环境搭建：

参考 `RAG环境搭建.md` 文档进行python环境的搭建

#### 基础流程：

```
1. 准备知识库
   编辑 docs.txt，每行一个文档

2. 构建索引
   python build_vector_index.py
   → 生成 indexes/vectors.faiss 和 indexes/chunks.pkl

3. 运行检索
   python rag_demo.py
   → 交互式查询知识库
```

### 更新知识库

更新数据库不仅需要在docs.txt中完成，还需要重构索引

```
1. 修改 docs.txt

2. 重新构建索引
   python build_vector_index.py

3. 继续使用检索程序
   python rag_demo.py
```

---

## 3. 参数说明

以下参数统一在 `config.py` 中维护。

### 文本分块参数

- `MAX_LENGTH`：单个文本块最大字符数。
- `OVERLAP`：相邻块重叠字符数。

### 路径参数

- `DOCS_FILE`：原始知识库文件路径。
- `INDEX_DIR`：索引目录。
- `INDEX_FILE`：FAISS 向量索引路径。
- `CHUNKS_FILE`：块文本与来源信息路径。

### 向量模型参数

- `VECTOR_INDEX_MODEL_NAME`：向量编码模型路径或名称。

### 查询控制参数

- `MAX_COUNT`：单次运行最多问答轮数。
- `TOP_K`：最终返回结果条数上限。

### 检索参数（FAISS + BM25 + rerank）

- `FAISS_CANDIDATE_K`：FAISS 初筛候选数。
- `BM25_CANDIDATE_K`：BM25 初筛候选数。
- `HYBRID_CANDIDATE_K`：融合后进入 rerank 的候选数。
- `RERANK_TOP_K`：单路 rerank 保留数量。
- `FAISS_SCORE_WEIGHT`：FAISS 融合权重。
- `BM25_SCORE_WEIGHT`：BM25 融合权重。

### LLM 选择参数

- `LLM_PROVIDER`：大模型提供方。
  - 可选值：`deepseek`、`qwen_local`

### DeepSeek 参数

- `DEEPSEEK_API_KEY`：DeepSeek API Key。
- `DEEPSEEK_BASE_URL`：DeepSeek API 地址。
- `LLM_MODEL_NAME`：DeepSeek 模型名。
- `DEEPSEEK_TIMEOUT`：接口超时秒数。

### Qwen 本地参数

- `QWEN_MODEL_NAME`：Qwen 显示名。
- `QWEN_LOCAL_PATH`：本地 Qwen 模型目录。
- `QWEN_MAX_NEW_TOKENS`：单次最大生成长度。
- `QWEN_TEMPERATURE`：采样温度。
- `QWEN_TOP_P`：核采样阈值。
- `QWEN_REPETITION_PENALTY`：重复惩罚系数。

### 输出参数

- `SHOW_BEST_ONLY`：
  - `True`：仅显示模型回答。
  - `False`：显示检索详情 + 模型回答。

### Prompt 工程参数（模型共享）

Prompt 不在 `config.py`，统一放在 `prompt_template.py`：

- `SYSTEM_PROMPT`：系统提示词。
- `USER_PROMPT_TEMPLATE`：用户模板。
- `build_user_prompt(query, contexts)`：统一构造 DeepSeek / Qwen 请求 Prompt。

当前模板约束为：

1. 只能基于上下文回答。
2. 找不到答案时输出“未找到相关信息”。
3. 不编造、可综合多条上下文、不要暴露上下文编号。

---

## 4. 文件说明

### config.py

- 程序结构：按“分块、路径、检索、LLM、输出”分组定义全局参数。
- 核心作用：统一控制系统行为，避免参数分散在各脚本中。
- 被调用关系：由 `rag_demo.py`、`build_vector_index.py`、`retrieve.py`、`deepseek_API.py`、`Qwen_API.py` 统一读取。

### chunker.py

- 程序结构：以 `chunk_text()` 为主入口，包含段落切分、句子切分、超长句窗口切分三层逻辑。
- 核心算法：段落优先 + 句子拼接 + overlap 重叠，平衡语义完整性与召回粒度。

### build_vector_index.py

- 程序结构：读取 `docs.txt` -> 调用 `chunker.py` -> 向量化 -> 构建 FAISS -> 保存到 `indexes/`。
- 核心算法：SentenceTransformer 编码 + FAISS `IndexFlatL2` 建索引。
- 调用关系：
   - 读取 `config.py` 参数；
   - 调用 `build_vector_index.py` 用于离线分块。
   - 产物供 `rag_demo.py` 运行时加载。

### retrieve.py

- 程序结构：`BM25Retriever`、`hybrid_retrieve()`、`hybrid_retrieve_with_query_split()` 三层检索接口。
- 核心算法流程：
   - BM25 关键词检索。
   - FAISS 向量检索。
   - 归一化加权融合（FAISS/BM25）。
   - 语义 rerank（余弦相似度）。
   - Query 拆分多路检索后合并去重。
- 调用关系：被 `rag_demo.py` 调用；读取 `config.py` 检索参数。

### rag_demo.py

- 程序结构：系统启动（加载向量模型与索引）-> 交互循环（检索）-> 生成回答（DeepSeek/Qwen）。
- 核心流程：
   - 调用 `retrieve.py` 输出 Top-K 上下文。
   - 按 `LLM_PROVIDER` 分流调用 `deepseek_API.py` 或 `Qwen_API.py`。
   - 处理输入输出，实现问答体系。
- 调用关系：是 **系统主入口** ，统一串联所有模块。

### deepseek_API.py

- 程序结构：封装 OpenAI 兼容接口调用 + DeepSeek 专用适配函数。
- 核心逻辑：将检索上下文和问题组装为统一 Prompt，发送 chat completions 请求。
- 调用关系：
- 读取 `prompt_template.py` 生成 Prompt。
- 被 `rag_demo.py` 在 `LLM_PROVIDER=deepseek` 时调用。

### Qwen_API.py

- 程序结构：本地模型懒加载/预加载 + 文本生成推理。
- 核心逻辑：首次加载 `qwen_local` 模型到设备，然后复用模型进行多轮问答。
- 调用关系：
- 读取 `prompt_template.py` 生成 Prompt。
- 被 `rag_demo.py` 在 `LLM_PROVIDER=qwen_local` 时调用。

### prompt_template.py

- 程序结构：`SYSTEM_PROMPT`、`USER_PROMPT_TEMPLATE`、`build_user_prompt()`。
- 核心作用：Prompt 工程中心化，改一处可同时影响 DeepSeek 与 Qwen。
- 调用关系：被 `deepseek_API.py` 与 `Qwen_API.py` 共同调用。

### download_bge_model.py

- 程序结构：下载并保存向量模型脚本。
- 核心作用：提前准备 `bge-small-zh/`，避免首次运行临时下载。
- 调用关系：手动执行一次即可，供 `build_vector_index.py` 与 `rag_demo.py` 使用。

### download_Qwen_model.py

- 程序结构：从 HuggingFace 下载 Qwen 模型并保存到本地目录。
- 核心作用：准备 `qwen_local/`，供本地生成模块加载。
- 调用关系：手动执行一次后，`Qwen_API.py` 通过 `QWEN_LOCAL_PATH` 调用。

### docs.txt

- 程序结构：纯文本知识库，每行一个文档单元。
- 核心作用：索引构建的原始语料来源。
- 调用关系：被 `build_vector_index.py` 读取。

### test_env.py

- 程序结构：环境检查脚本。
- 核心作用：验证 PyTorch、CUDA、向量模型加载是否正常。
- 调用关系：独立手动执行，不参与主流程运行。

### bge-small-zh/

- 目录作用：本地向量编码模型目录。
- 主流程关系：`build_vector_index.py` 与 `rag_demo.py` 在编码阶段读取该目录模型。

### indexes/

- 目录作用：离线索引与块元数据目录。
- 主流程关系：
- `build_vector_index.py` 负责写入。
- `rag_demo.py` 负责读取。

### qwen_local/

- 目录作用：本地 Qwen 生成模型目录。
- 主流程关系：`Qwen_API.py` 按 `QWEN_LOCAL_PATH` 加载该目录进行回答生成。
