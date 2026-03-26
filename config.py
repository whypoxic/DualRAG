"""
RAG 系统全局配置文件
统一管理所有参数和路径
"""

# ===== 文本分块配置 =====
MAX_LENGTH = 256  # 临界点：块的最大字符数。超过此值才触发二级拆分。
OVERLAP = 50      # 重叠字符数，定制为 MAX_LENGTH 的 20%。

# ===== 文件路径 =====
DOCS_FILE = "./docs.txt"              # 原始知识库文件（每行一个文档）
INDEX_DIR = "./indexes"               # 索引存储目录
INDEX_FILE = "./indexes/vectors.faiss"  # FAISS 索引文件路径
CHUNKS_FILE = "./indexes/chunks.pkl"  # 块文本和源信息文件路径

# ===== 模型配置 =====
VECTOR_INDEX_MODEL_NAME = "./bge-small-zh"      # 向量编码模型名称

# ===== 查询配置 =====
MAX_COUNT = 3       # 最多允许的查询次数，防止无限循环
TOP_K = 5         # 检索返回的块数量

# ===== 检索配置（FAISS + BM25 + Rerank） =====
FAISS_CANDIDATE_K = 20  # FAISS 初筛候选数量
BM25_CANDIDATE_K = 20   # BM25 初筛候选数量
HYBRID_CANDIDATE_K = 30  # 融合后进入 rerank 的候选数量
RERANK_TOP_K = 5         # rerank 后保留的最终结果数量

# 分数融合权重（建议和为 1.0）
FAISS_SCORE_WEIGHT = 0.6
BM25_SCORE_WEIGHT = 0.4

# ===== LLM 选择配置 =====
# 可选值："deepseek" 或 "qwen_local"
LLM_PROVIDER = "qwen_local"

# ===== DeepSeek 配置（OpenAI 兼容接口） =====
DEEPSEEK_API_KEY = "" # 请替换为你的 DeepSeek API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LLM_MODEL_NAME = "deepseek-chat"
DEEPSEEK_TIMEOUT = 60

# ===== Qwen 本地模型配置 =====
QWEN_MODEL_NAME = "qwen_local"
QWEN_LOCAL_PATH = "./qwen_local"
QWEN_MAX_NEW_TOKENS = 512
QWEN_TEMPERATURE = 0.2
QWEN_TOP_P = 0.9
QWEN_REPETITION_PENALTY = 1.05

# ===== 输出配置 =====
SHOW_BEST_ONLY = True  # 是否仅显示AI模型输出结果
