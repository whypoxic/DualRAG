"""
deepseek_API.py
统一管理 LLM 调用接口，当前提供 DeepSeek 适配。

后续可在此基础上扩展其他 OpenAI 兼容供应商，
"""

from typing import List, Optional
import time

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    LLM_MODEL_NAME,
    DEEPSEEK_TIMEOUT,
)
# 从 prompt_template 导入系统提示和用户提示构建函数
from prompt_template import SYSTEM_PROMPT, build_user_prompt
from logger_setup import get_logger


logger = get_logger("llm.deepseek", "llm.log")


def generate_with_openai_compatible(
    query: str,
    contexts: List[str],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 60,
    system_prompt: Optional[str] = None,
) -> Optional[str]:
    """
    通用 OpenAI 兼容接口调用。
    适用于 DeepSeek 以及其他兼容 OpenAI Chat Completions 的服务。
    """

    if not api_key.strip():
        logger.warning("OpenAI兼容调用失败: API Key为空")
        return None

    start = time.perf_counter()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    final_system_prompt = system_prompt or SYSTEM_PROMPT
    user_prompt = build_user_prompt(query=query, contexts=contexts)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": final_system_prompt},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        stream=False,
    )
    logger.info(
        "OpenAI兼容调用成功: model=%s, contexts=%d, elapsed_ms=%.2f",
        model,
        len(contexts),
        (time.perf_counter() - start) * 1000,
    )
    return response.choices[0].message.content.strip()


def generate_with_deepseek(query: str, contexts: List[str]) -> Optional[str]:
    """
    DeepSeek 专用适配器，对外暴露稳定函数名。
    """
    if not DEEPSEEK_API_KEY.strip():
        logger.warning("DeepSeek调用跳过: API Key未配置")
        print("\nDeepSeek API Key 未配置，跳过生成回答。")
        print("请在 config.py 中填写 DEEPSEEK_API_KEY。")
        return None

    try:
        logger.info("开始DeepSeek调用: model=%s, query_len=%d, contexts=%d", LLM_MODEL_NAME, len(query), len(contexts))
        return generate_with_openai_compatible(
            query=query,
            contexts=contexts,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model=LLM_MODEL_NAME,
            timeout=DEEPSEEK_TIMEOUT,
        )
    except Exception as e:
        logger.exception("DeepSeek调用失败")
        print(f"\nDeepSeek 调用失败: {e}")
        return None
