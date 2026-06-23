from config import TEXT_SPLIT_CONFIG


def split_text(text: str) -> list[str]:
    """文本分割：按指定长度切分，保留重叠上下文"""
    chunk_size = TEXT_SPLIT_CONFIG["chunk_size"]
    chunk_overlap = TEXT_SPLIT_CONFIG["chunk_overlap"]
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap  # 重叠部分
        if start >= text_len - chunk_overlap:
            break
    # 补充最后一段（避免遗漏）
    if start < text_len:
        chunks.append(text[start:])
    return chunks


# 测试：读取本地文档并分割
def load_and_split_doc(doc_path: str) -> list[str]:
    with open(doc_path, "r", encoding="utf-8") as f:
        doc_text = f.read()
    return split_text(doc_text)