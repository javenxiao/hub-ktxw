import re
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextProcessor:
    """文本处理模块 - 处理解析后的文本内容"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """文本清洗预处理"""
        # 移除多余空行和空格
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # 移除特殊的Markdown格式字符但保留基本结构
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # 移除图片标记但保留文字
        text = re.sub(r'#{1,6}\s*', '', text)  # 移除标题标记但保留标题文字

        return text.strip()

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        分割文本为块

        Args:
            text: 输入文本

        Returns:
            分割后的文本块列表
        """
        cleaned_text = self.clean_text(text)

        # 使用LangChain的文本分割器
        chunks = self.text_splitter.split_text(cleaned_text)

        # 为每个块添加元数据
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # 过滤掉太短的块
                processed_chunks.append({
                    "id": i,
                    "content": chunk.strip(),
                    "length": len(chunk),
                    "token_count": len(chunk.split())  # 简化的token计数
                })

        self.logger.info(f"文本分割完成，共 {len(processed_chunks)} 个块")
        return processed_chunks

    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """验证分割质量"""
        if not chunks:
            return False

        # 检查块大小分布
        sizes = [chunk['length'] for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)

        # 验证平均大小在合理范围内
        return 500 <= avg_size <= 1500