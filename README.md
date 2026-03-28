# RAG 项目说明

这是一个可本地运行的 RAG 检索增强生成项目，支持：

- 双模型回答通道：DeepSeek API 或本地 Qwen

- 向量数据集构建
- 向量检索（FAISS）+ 关键词检索（BM25）
- 混合召回 + rerank
- 多问题 Query 拆分检索
- 统一 Prompt 工程

用于：

- 实践 RAG 基本流程
- 快速搭建可迭代的小型知识问答系统
- 对比云端模型与本地模型的接入方式

## 项目做了什么

完整流程如下：

1. 读取知识库文本（docs.txt）
2. 分块（chunker.py）
3. 编码并离线构建索引（build_vector_index.py）
4. 查询时执行混合检索（retrieve.py）
5. 将 Top-K 上下文交给大模型生成答案（deepseek_API.py 或 Qwen_API.py）

## 项目结构（核心）

- [rag_demo.py](rag_demo.py)：主程序入口
- [config.py](config.py)：全局配置
- [retrieve.py](retrieve.py)：FAISS + BM25 + rerank + Query 拆分
- [build_vector_index.py](build_vector_index.py)：离线索引构建
- [chunker.py](chunker.py)：文本分块
- [deepseek_API.py](deepseek_API.py)：DeepSeek 调用
- [Qwen_API.py](Qwen_API.py)：本地 Qwen 调用
- [prompt_template.py](prompt_template.py)：共享 Prompt 
- [logger_setup.py](logger_setup.py)：统一日志初始化
- [docs.txt](docs.txt)：知识库
- [indexes/](indexes/)：索引目录
- [log/](log/)：运行日志目录
- [bge-small-zh/](bge-small-zh/)：向量模型目录
- [qwen_local/](qwen_local/)：本地 Qwen 模型目录

## 第一次接触这个项目，如何操作

### 第 0 步：推荐先看这两份文档

- [RAG运行说明.md](RAG运行说明.md)
- [RAG环境搭建.md](RAG环境搭建.md)

### 第 1 步：准备 Python 环境

建议使用 Python 3.10。

安装常用依赖（按需执行）：

```bash
pip install torch torchvision
pip install faiss-cpu
pip install sentence-transformers
pip install transformers
pip install numpy
pip install openai
```

更完整环境说明，请看：

- [RAG环境搭建.md](RAG环境搭建.md)

### 第 2 步：准备模型

模型下载先参考：  
- [RAG环境搭建.md](RAG环境搭建.md)

向量模型：

- 若项目中已有 bge-small-zh 目录，可直接使用
- 若没有，可运行下载脚本：
  python download_bge_model.py

Qwen 本地模型（可选）：

- 若你要用本地 Qwen，确保存在 qwen_local 目录安装自己适配的Qwen模型；
- 若没有，可运行下载脚本：
  python download_Qwen_model.py

### 第 3 步：配置参数

打开 [config.py](config.py)，至少确认以下项：

- VECTOR_INDEX_MODEL_NAME
- DOCS_FILE
- LLM_PROVIDER（deepseek 或 qwen_local）

如果使用 DeepSeek：

- 填写 DEEPSEEK_API_KEY
- 检查 DEEPSEEK_BASE_URL 和 LLM_MODEL_NAME

如果使用本地 Qwen：

- 设置 LLM_PROVIDER = qwen_local
- 检查 QWEN_LOCAL_PATH

### 第 4 步：准备知识库

编辑 [docs.txt](docs.txt)，每行一条知识文本。

请根据所需要的知识库调整文本内容；或采用其他格式的知识库（需要调整接口）

### 第 5 步：先构建索引

```bash
python build_vector_index.py
```

执行后会生成：

- indexes/vectors.faiss
- indexes/chunks.pkl

### 第 6 步：启动问答

```bash
python rag_demo.py
```

然后按提示输入问题即可。

## 常见工作方式

更新知识库后：

1. 修改 docs.txt
2. 重新运行 build_vector_index.py
3. 再运行 rag_demo.py

调整 Prompt 后：

- 修改 [prompt_template.py](prompt_template.py)
- DeepSeek 与 Qwen 会同步使用新 Prompt

## 日志模块

- 日志统一通过 [logger_setup.py](logger_setup.py) 初始化，默认写入 [log/](log/)。
- 运行主程序 [rag_demo.py](rag_demo.py) 时，会自动记录主流程、检索与模型调用日志。
- 运行索引构建 [build_vector_index.py](build_vector_index.py) 时，会自动记录构建阶段日志。
- 日志模块不需要单独启动；业务脚本运行时会自动生效。
- 详细说明可查看 [日志说明.md](日志说明.md)。

