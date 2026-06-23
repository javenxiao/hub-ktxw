import hashlib
import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss

from sentence_transformers import SentenceTransformer


class SemanticRouter:

    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            embedding_dim: int,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = "123456",
            distance_threshold=0.5
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        self.embedding_dim = embedding_dim
        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None

    def add_route(self, questions: List[str], target: str):
        # Embedding
        question_vectors = self.embedding_method(questions)
        # 向量保存
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(question_vectors)
        # 遍历处理
        for i, question in enumerate(questions):
            # Redis保存
            try:
                with self.redis.pipeline() as pipe:
                    pipe.set(self.name + "key:" + question, target)
                    pipe.rpush(self.name + "list", question)
                    pipe.execute()
            except:
                import traceback
                traceback.print_exc()
                return -1
        faiss.write_index(self.index, f"{self.name}.index")

    def route(self, question: str):
        if self.index is None:
            return None
        # 新的提问进行编码
        embedding = self.embedding_method(question)
        # 向量数据库中进行检索
        dis, ind = self.index.search(embedding, k=1)
        if dis[0][0] > self.distance_threshold:
            return None
        index = ind[0][0]
        questions = self.redis.lrange(self.name + "list", 0, -1)
        # 匹配问题
        match_question = questions[index]
        print('match_question ->', match_question)
        # 匹配结果
        return self.redis.get(self.name + "key:" + match_question.decode()).decode()


if __name__ == "__main__":
    model = SentenceTransformer('../../../models/BAAI/bge-small-zh-v1.5')

    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        return np.array([model.encode(t) for t in text])

    router = SemanticRouter(
        name="semantic_route",
        embedding_method=get_embedding,
        embedding_dim=model.get_sentence_embedding_dimension(),
    )
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    question = "Hi, good morning"
    target = router.route(question)
    print(question, " ->", target)