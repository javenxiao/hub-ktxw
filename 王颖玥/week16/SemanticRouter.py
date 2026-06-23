from typing import Optional, List, Union, Callable, Any
import faiss
from redisvl.extensions.router import Route, SemanticRouter
from sentence_transformers import SentenceTransformer
import redis
import json
import os
import numpy as np

model = SentenceTransformer("/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-small-zh-v1.5")

class SemanticRouter:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.1
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        # 初始化Faiss索引
        if os.path.exists(f"{self.name}.index"):  # 检查是否存在Faiss索引文件
            self.index = faiss.read_index(f"{self.name}.index")   # 加载已有索引
        else:
            self.index = None  # 无索引文件则初始化索引为None

    def add_route(self, questions: List[str], target: str):
        if not questions:
            return

        question_vectors = self.embedding_method(questions)
        vec_dim = question_vectors.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(vec_dim)

        start_idx = self.index.ntotal
        self.index.add(question_vectors)
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for i, q in enumerate(questions):
                    idx = start_idx + i
                    pipe.setex(f"{self.name}idx:{idx}", self.ttl, target)
                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1

    def route(self, question: str):
        query_vector = self.embedding_method(question)

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        if self.index is None:
            return None

        distances, indices = self.index.search(query_vector, k=1)

        if distances[0][0] > self.distance_threshold:
            return None

        target = self.redis.get(f"{self.name}idx:{indices[0][0]}")
        return target

    def __call__(self, query: str):
        return self.route(query)

    def clear_cache(self):
        for key in self.redis.scan_iter(f"{self.name}*"):
            self.redis.delete(key)

        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")   # 删除Faiss索引文件
        self.index = None  # 重置索引为None


if __name__ == "__main__":
    def get_embedding(text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]

        embeddings = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings

    router = SemanticRouter(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        distance_threshold=0.5
    )
    router.clear_cache()

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )

    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    print("Hello! Morning: ", router("Hello! Morning"))
    print("退货怎么操作？:", router("退货怎么操作？"))
    print("今天心情很好:", router("今天心情很好"))
