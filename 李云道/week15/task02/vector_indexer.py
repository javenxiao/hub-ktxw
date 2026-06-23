import os
import pickle
import logging
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer


class VectorIndexer:
    """向量索引模块 - 使用FAISS构建向量数据库"""

    def __init__(self, model_name: str = r"D:\Download\BAAI\bge-small-zh-v1.5"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.logger = logging.getLogger(__name__)

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """创建文本嵌入向量"""
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        return np.array(embeddings).astype('float32')

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        构建FAISS向量索引

        Args:
            chunks: 文本块列表
        """
        if not chunks:
            raise ValueError("无法为空文本块构建索引")

        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)

        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度

        # 标准化向量以便使用内积相似度
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        self.logger.info(f"FAISS索引构建完成，维度: {dimension}, 向量数: {len(chunks)}")

    def search_similar(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回最相似的k个结果

        Returns:
            (文本块, 相似度得分) 的列表
        """
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用build_index")

        # 生成查询向量
        query_embedding = np.array([self.embedder.embed_query(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # 搜索相似向量
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):  # 确保索引有效
                results.append((self.chunks[idx], float(score)))

        return results

    def save_index(self, file_path: str) -> None:
        """保存索引到文件"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(self.index),
                'chunks': self.chunks,
                'model_name': self.model_name
            }, f)

    def load_index(self, file_path: str) -> None:
        """从文件加载索引"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.chunks = data['chunks']
            self.model_name = data['model_name']
