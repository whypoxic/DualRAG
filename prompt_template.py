"""
prompt_template.py
统一管理 DeepSeek / Qwen 共用的 Prompt 模板。
"""

from typing import List


SYSTEM_PROMPT = "你是一个严谨的AI助手，请严格根据提供的上下文回答问题："

USER_PROMPT_TEMPLATE = """
【要求】
1. 只能基于上下文回答
2. 如果找不到答案，说“未找到相关信息”
3. 不要编造任何未提供的信息
4. 如果上下文中有多个相关信息，可以综合使用，但不要简单罗列
5. 不要在回答中提及上下文的编号或格式

【上下文】
{context}

【问题】
{query}
"""


def build_context_text(contexts: List[str]) -> str:
    return "\n\n".join([f"[{i + 1}] {text}" for i, text in enumerate(contexts)])


def build_user_prompt(query: str, contexts: List[str]) -> str:
    context_text = build_context_text(contexts)
    return USER_PROMPT_TEMPLATE.format(context=context_text, query=query)
