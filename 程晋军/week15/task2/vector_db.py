import faiss
import numpy as np
import os
from config import VECTOR_DB_CONFIG, EMBEDDING_CONFIG
from embedding_generator import EmbeddingGenerator  # 导入改为新类名

class FaissVectorDB:
    def __init__(self):
        self.embedding_dim = EMBEDDING_CONFIG["embedding_dim"]
        self.db_path = VECTOR_DB_CONFIG["db_path"]
        self.top_k = VECTOR_DB_CONFIG["top_k"]
        self.embedding_generator = EmbeddingGenerator()  # 实例化新类
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.text_chunks = []

    # 后续build_db、load_db、search方法完全不变，直接复用
    def build_db(self, text_chunks: list[str]):
        self.text_chunks = text_chunks
        embeddings = self.embedding_generator.batch_get_embedding(text_chunks)
        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)
        os.makedirs(self.db_path, exist_ok=True)
        faiss.write_index(self.index, f"{self.db_path}/faiss_index.index")
        with open(f"{self.db_path}/text_chunks.txt", "w", encoding="utf-8") as f:
            f.write("\n===CHUNK_SPLITTER===\n".join(text_chunks))
        print(f"向量库构建完成，共{len(text_chunks)}个文本块")

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError("向量库不存在，请先构建")
        self.index = faiss.read_index(f"{self.db_path}/faiss_index.index")
        with open(f"{self.db_path}/text_chunks.txt", "r", encoding="utf-8") as f:
            self.text_chunks = f.read().split("\n===CHUNK_SPLITTER===\n")
        print(f"向量库加载完成，共{len(self.text_chunks)}个文本块")

    def search(self, query: str) -> list[str]:
        query_embedding = self.embedding_generator.get_embedding(query)
        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_np, self.top_k)
        relevant_chunks = [self.text_chunks[idx] for idx in indices[0] if idx < len(self.text_chunks)]
        return relevant_chunks

if __name__ == "__main__":
    # 简单测试向量库功能
    vector_db = FaissVectorDB()
    sample_texts = [
        "这是第一段测试文本。",
        "这是第二段测试文本，内容稍微长一些以测试分割效果。",
        "第三段文本在这里。",
        "第四段文本包含更多的信息，用于测试向量检索的准确性。",
        "最后一段测试文本。"
    ]
    vector_db.build_db(sample_texts)
    vector_db.load_db()
    query = "测试文本内容"
    results = vector_db.search(query)
    print("检索结果：")
    for res in results:
        print(res)