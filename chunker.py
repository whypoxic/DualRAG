# chunker.py
# 这是一个文本分块模块，提供分层的文本切割功能（段落→句子→字符窗口）。

import re

# 全局配置
MAX_LENGTH = 256  # 临界点——块的最大字符数。超过此值才触发二级拆分。影响块的粒度和检索精度。
OVERLAP = 50  # 重叠字符数，约为 MAX_LENGTH 的 20%。用于在块间保留上下文。


def chunk_text(text, max_length=MAX_LENGTH, overlap=OVERLAP):
	"""
	分层拆分文本：
	1. 第一层：按段落分割（\n\n）— 保留原始段落结构
	2. 第二层：若段落超长，按句子切割（。！？.等）— 保留语义完整性
	3. 第三层：若单句超长，按字符窗口硬切 — 不得已的最后手段
	
	Args:
		text (str): 输入文本
		max_length (int): 临界值，单个块的目标最大长度（字符数）
		overlap (int): 块间重叠长度，避免上下文丢失
	
	Returns:
		list: 分块后的文本列表
	"""
	text = text.strip()
	if not text:
		return []

	# ===== 第一层：按段落分割 =====
	paragraphs = text.split('\n\n')
	all_chunks = []

	for para in paragraphs:
		para = para.strip()
		if not para:
			continue

		# 段落很短，直接放入
		if len(para) <= max_length:
			all_chunks.append(para)
			continue

		# ===== 第二层：段落超长，按句子切割 =====
		# 用正则识别中英文句子边界
		sentences = re.split(r'([。！？!?；;..])', para)
		sentence_list = []
		for i in range(0, len(sentences), 2):
			sent = sentences[i].strip()
			if not sent:
				continue
			# 获取句子末尾的标点
			punct = sentences[i + 1] if i + 1 < len(sentences) else ""
			sentence_list.append(sent + punct)

		# 贪心合并：多个句子组成一个块，直到超过 max_length
		current_chunk = ""
		for sent in sentence_list:
			# ===== 第三层：单句超长，才硬切 =====
			if len(sent) > max_length:
				# 先把已积累的块输出
				if current_chunk:
					all_chunks.append(current_chunk)
					current_chunk = ""
				# 对超长句做字符级窗口切割
				start = 0
				step = max(1, max_length - overlap)
				while start < len(sent):
					chunk = sent[start : start + max_length]
					if chunk.strip():
						all_chunks.append(chunk)
					start += step
				continue

			# 正常情况：句子长度合理，尝试合并
			if len(current_chunk) + len(sent) <= max_length:
				current_chunk += sent
			else:
				# 当前块已满，保存并开启新块（带重叠）
				if current_chunk:
					all_chunks.append(current_chunk)
					# 重叠：保留当前块的尾部，作为下一块的开头
					tail = current_chunk[-overlap:] if overlap > 0 else ""
					current_chunk = tail + sent
				else:
					current_chunk = sent

		# 剩余内容入块
		if current_chunk:
			all_chunks.append(current_chunk)

	return all_chunks
