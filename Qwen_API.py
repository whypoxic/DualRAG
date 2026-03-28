"""
Qwen_API.py
统一管理 Qwen 本地模型调用。
"""

from typing import List, Optional
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    QWEN_LOCAL_PATH,
    QWEN_MAX_NEW_TOKENS,
    QWEN_TEMPERATURE,
    QWEN_TOP_P,
    QWEN_REPETITION_PENALTY,
)
# 从 prompt_template 导入系统提示和用户提示构建函数
from prompt_template import SYSTEM_PROMPT, build_user_prompt
from logger_setup import get_logger


logger = get_logger("llm.qwen", "llm.log")


_tokenizer = None
_model = None
_model_path = None


def preload_qwen_model() -> bool:
    """
    启动阶段提前加载本地 Qwen 模型，减少首问延迟。
    """
    try:
        start = time.perf_counter()
        _load_model_if_needed(QWEN_LOCAL_PATH)
        logger.info("Qwen预加载成功: model_path=%s, elapsed_ms=%.2f", QWEN_LOCAL_PATH, (time.perf_counter() - start) * 1000)
        print(f"Qwen 模型已预加载: {QWEN_LOCAL_PATH}")
        return True
    except Exception as e:
        logger.exception("Qwen预加载失败")
        print(f"Qwen 预加载失败: {e}")
        return False


def _load_model_if_needed(model_path: str):
    global _tokenizer, _model, _model_path

    if _model is not None and _tokenizer is not None and _model_path == model_path:
        logger.debug("Qwen模型已缓存: model_path=%s", model_path)
        return _tokenizer, _model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    load_start = time.perf_counter()
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if device == "cuda":
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
        )
        _model = _model.to(device)

    _model.eval()
    _model_path = model_path
    logger.info("Qwen模型加载完成: model_path=%s, device=%s, elapsed_ms=%.2f", model_path, device, (time.perf_counter() - load_start) * 1000)

    return _tokenizer, _model


def generate_with_qwen(query: str, contexts: List[str]) -> Optional[str]:
    """
    使用本地 Qwen 模型生成回答。
    """
    try:
        start = time.perf_counter()
        logger.info("开始Qwen生成: query_len=%d, contexts=%d", len(query), len(contexts))
        tokenizer, model = _load_model_if_needed(QWEN_LOCAL_PATH)

        user_prompt = build_user_prompt(query=query, contexts=contexts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"系统: {SYSTEM_PROMPT}\n用户: {user_prompt}\n助手:"

        inputs = tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = QWEN_TEMPERATURE > 0
        temperature = QWEN_TEMPERATURE if QWEN_TEMPERATURE > 0 else 1.0

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=temperature,
                top_p=QWEN_TOP_P,
                repetition_penalty=QWEN_REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_len:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.info("Qwen生成完成: answer_ok=%s, elapsed_ms=%.2f", bool(answer), (time.perf_counter() - start) * 1000)

        return answer if answer else None

    except Exception as e:
        logger.exception("Qwen本地模型调用失败")
        print(f"\nQwen 本地模型调用失败: {e}")
        return None
