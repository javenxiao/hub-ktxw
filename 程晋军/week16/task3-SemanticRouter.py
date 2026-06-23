from typing import Optional, List, Union, Any, Dict
import redis

class SemanticRouter:
    def __init__(
            self,
            name: str,  # 对话的名字，类似session id
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl

    def add_route(self, questions: List[str], target: str):
        if not questions:  # 新增：空列表防护，避免无效循环
            raise ValueError("问题列表(questions)不能为空")
        for question in questions:
            self.redis.setex(question, self.ttl,target)



    def route(self, question: str):
        if not question:  # 新增：空列表防护，避免无效循环
            raise ValueError("问题不能为空")
        target=self.redis.get(question)
        print(f"Routing question: {target}")


if __name__ == "__main__":
    router = SemanticRouter(
        name="my-session",
        # redis_url="localhost",
    )
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    router.route("如何退货")