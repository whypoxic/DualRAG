# RAG架构搭建手册

这里记录了我的RAG搭建  
包括
- 环境依赖
- 运行测试

- 模型安装或调用

## 环境搭建

首先，我们需要从一个纯净的py环境开始：

这里使用了 `conda` 环境

```bash
conda create -n rag python=3.10
conda activate rag
```

使用py3.10版本，适应性比较不错

---

接下来所有操作将在这个独立环境下进行：

### 安装`pytorch`:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
特殊说明：  
需要按照自己电脑的cuda版本安装对应版本的pytorch；  
可以用以下命令查看自己显卡的cuda版本
```bash
nvidia-smi
```
然后去pytorch官网查看安装连接
```
https://pytorch.org/get-started/locally/
```

### 安装`faiss`:
```
pip install faiss-cpu
```
特殊说明：  
window系统下只能使用faiss的cpu版本；需要在linux系统下才能使用gpu版本

### 安装其他依赖：
```
pip install sentence-transformers
pip install transformers
pip install tqdm
pip install numpy
```

### 安装 `bge-small-zh` 向量模型：

本项目默认使用本地目录 `./bge-small-zh` 作为向量模型路径。

可直接运行已提供脚本下载：

```bash
python download_bge_model.py
```

下载完成后会在项目根目录生成（或更新）`bge-small-zh/`，
`build_vector_index.py` 与 `rag_demo.py` 会通过 `config.py` 中的
`VECTOR_INDEX_MODEL_NAME` 自动加载该模型。

### 验证：

编辑一个测试文件 `test_env.py` 
```py
from sentence_transformers import SentenceTransformer
import torch
import faiss

print("fine!")
print("CUDA:", torch.cuda.is_available())

model = SentenceTransformer("BAAI/bge-small-zh", device="cuda")
vec = model.encode("RAG system test")

print("vec length:", len(vec))
```
运行过程可能会比较慢；因为模型要从HuggingFace官网上下载  

当能输出一下信息，说明环境搭建完成了：
```
fine!
CUDA: True
……
vec length: 512
```
---



## deepseek接口搭建

需要前往`https://platform.deepseek.com/`注册一个API_key

随后前往`config.py`中填写你的API_key

deepseek采用openai实现的接口：

```bash
pip install openai
```

具体实现已经也写好；安装相应库后可以直接运行demo程序；

---

以下是deepseek的官方示例文档，如需修改代码可以参考：
```py
# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'), # 修改自己的key
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```

## Qwen本地接口搭建

需要额外安装：
```bash
pip install accelerate
```

如果要下载到本地目录，可通过已经提供好的脚本：

`download_Qwen_model.py` 的作用：
1. 从 HuggingFace 下载 `Qwen/Qwen2.5-0.5B-Instruct`（最小模型，可以根据需求更换）
2. 自动保存到本地目录 `./qwen_local` （请手动创建一个新的文件夹用于存放）

脚本中已预留镜像和 token 说明（按需设置）：

```bash
set HF_TOKEN=你的token
set HF_ENDPOINT=https://hf-mirror.com
```

运行脚本：

```bash
python download_Qwen_model.py
```

下载完成后，在 `config.py` 中确认：

```python
LLM_PROVIDER = "qwen_local"
QWEN_LOCAL_PATH = "./qwen_local"
```

这样 `rag_demo.py` 就会走本地千问模型生成回答。