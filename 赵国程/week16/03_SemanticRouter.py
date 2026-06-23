from typing import Optional, List, Union, Any, Dict
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss

class SemanticRouter:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int=3600*24, # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.1
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None

    def add_route(self, questions: List[str], target: str):
        questions_vector = self.embedding_method(questions)
        if self.index is None:
            self.index = faiss.IndexFlatL2(questions_vector.shape[1])

        self.index.add(questions_vector)
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for q in questions:
                    pipe.setex(self.name + "key:" + q, self.ttl, target) # 提问和回答存储在redis
                    pipe.rpush(self.name + "list", q) #所有的提问都存储在list里面，方便后续使用
                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1

    def route(self, question: str):
        question_vector = self.embedding_method(question)
        if self.index is None:
            return None

        D, I = self.index.search(question_vector, 1)
        if D[0][0] > self.distance_threshold:
            return None

        all_questions = self.redis.lrange(self.name + "list", 0, -1)
        question = all_questions[I[0][0]].decode("utf-8")

        return self.redis.get(self.name + "key:" + question).decode("utf-8")


if __name__ == "__main__":
    router = SemanticRouter()
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    router("Hi, good morning")