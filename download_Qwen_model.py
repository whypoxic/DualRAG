from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 在运行前请确保已设置环境变量 HF_TOKEN 
# set HF_TOKEN=你的 Hugging Face 访问令牌

# set HF_ENDPOINT=https://hf-mirror.com

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 保存到本地
tokenizer.save_pretrained("./qwen_local")
model.save_pretrained("./qwen_local")

print("下载完成！")