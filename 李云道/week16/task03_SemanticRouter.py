import hashlib
import os
import random
import time
from typing import Optional, List, Union, Any, Dict

import faiss
import numpy as np
import redis
from pyarrow import float32
from sentence_transformers import SentenceTransformer
import torch


class SemanticRouter:
    model = SentenceTransformer(r'D:\Download\Qwen\Qwen3-Embedding-0.6B')
    if torch.cuda.is_available():
        model = model.cuda()

    def __init__(
            self,
            name: str = "cache:semantic",
            redis_host: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            ttl: int = 3600 * 24 * 7,
            ttl_offset: float = 0.25,
            similarity_threshold=0.5
    ):
        self.name = name if name.startswith("cache:") else "cache:" + name
        self.redis = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
        self.ttl = ttl
        self.ttl_offset = ttl_offset
        self.similarity_threshold = similarity_threshold
        self._build_index()

    def add_route(self, questions: List[str], target: str):

        # embedding
        embedding = SemanticRouter.get_embedding(questions)
        # md5
        questions_md5 = [SemanticRouter.encode_text_md5(q) for q in questions]
        # index
        start_id = self.index.ntotal  # 添加前的总数
        try:
            self.index.add(embedding)
            faiss.write_index(self.index, self.index_file_path)
        except Exception as e:
            import traceback
            traceback.print_exc()

        # redis
        ttl = self._get_ttl()
        with self.redis.pipeline() as pipe:
            # 元数据
            for i, q in enumerate(questions):
                # faiss_id -> questions_md5
                faiss_id = start_id + i
                pipe.set(f"{self.name}:faiss_id_to_md5:{faiss_id}", questions_md5[i])
                pipe.expire(f"{self.name}:faiss_id_to_md5:{faiss_id}", ttl)

                # questions_md5 -> faiss_id
                pipe.set(f"{self.name}:md5_to_faiss_id:{questions_md5[i]}", faiss_id)
                pipe.expire(f"{self.name}:md5_to_faiss_id:{questions_md5[i]}", ttl)

                # 正向索引 questions_md5->target
                pipe.hset(f"{self.name}:question:{questions_md5[i]}", mapping={
                    "target": target,
                    "text": q,
                    "create_time": time.time(),
                    "hit_count": 0

                })
                pipe.expire(f"{self.name}:question:{questions_md5[i]}", ttl)
            # 反向索引 target->questions_md5
            pipe.sadd(f"{self.name}:target", *questions_md5)
            pipe.expire(f"{self.name}:target", ttl)

            # 添加路由
            pipe.sadd(f"{self.name}:route", target)
            pipe.expire(f"{self.name}:route", ttl)

            pipe.smembers(f"{self.name}:target")
            result = pipe.execute()
            return result[-1]

    def route(self, question: str, k=3):
        # 1. 计算问题的MD5用于缓存查询
        question_md5 = SemanticRouter.encode_text_md5(question)

        # 2. 先检查Redis精确匹配（相同问题缓存）
        cached_result = self.redis.hgetall(f"{self.name}:question:{question_md5}")
        if cached_result:
            # 更新命中次数
            self.redis.hincrby(f"{self.name}:question:{question_md5}", "hit_count", 1)
            return cached_result[b"target"].decode() if isinstance(cached_result.get(b"target"), bytes) else \
                cached_result["target"]

        # 3. 缓存未命中，进行向量化搜索
        embedding = self.get_embedding([question])

        # 4. FAISS搜索，获取k个最近邻
        distances, indices = self.index.search(embedding, k=k)

        # 5. 检查距离阈值
        if distances[0][0] < self.similarity_threshold:
            return None

        # 6. 从Redis获取相似问题的target
        target_candidates = []
        for i in range(k):
            if distances[0][i] < self.similarity_threshold:
                continue

            # 获取相似问题的MD5
            # 需要建立FAISS索引ID到question_md5的映射
            # 可以在add_route时存储这个映射
            faiss_id = indices[0][i]
            similar_md5 = self.redis.get(f"{self.name}:faiss_id_to_md5:{faiss_id}")

            if similar_md5:
                similar_md5 = similar_md5.decode() if isinstance(similar_md5, bytes) else similar_md5
                similar_info = self.redis.hgetall(f"{self.name}:question:{similar_md5}")
                if similar_info:
                    target = similar_info[b"target"].decode() if isinstance(similar_info.get(b"target"), bytes) else \
                        similar_info["target"]
                    target_candidates.append((target, distances[0][i]))

        if not target_candidates:
            return None

        # 7. 选择最佳匹配（距离最小的）
        target_candidates.sort(key=lambda x: x[1])
        best_target = target_candidates[0][0]

        # 8. 将新问题加入缓存
        # self.redis.hsetex(
        #     f"{self.name}:question:{question_md5}",
        #     mapping={
        #         "target": best_target,
        #         "text": question,
        #         "create_time": time.time(),
        #         "hit_count": 1
        #     },
        #     ex=self._get_ttl()
        # )

        # 9. 将问题添加到目标集合
        # self.redis.sadd(f"{self.name}:target:{best_target}", question_md5)

        return best_target

    def __call__(self, *args, **kwargs):
        return self.route(*args, **kwargs)

    @staticmethod
    def encode_text_md5(text: str):
        return hashlib.md5(text.encode()).hexdigest()

    def _get_ttl(self):
        return self.ttl + int(self.ttl * random.random() * self.ttl_offset)

    @staticmethod
    def get_embedding(text: list[str]):
        embedding = SemanticRouter.model.encode(text, normalize_embeddings=True)
        if isinstance(embedding, torch.Tensor):
            return embedding.numpy()
        elif isinstance(embedding, np.ndarray):
            return embedding
        else:
            return np.array(embedding, dtype=np.float32)

    def _build_index(self):
        self.index_file_path = f"{self.name.split(":")[0]}.index"
        if os.path.exists(self.index_file_path):
            try:
                self.index = faiss.read_index(self.index_file_path)
                return
            except Exception as e:
                import traceback
                traceback.print_exc()

        embed = SemanticRouter.get_embedding([self.index_file_path])
        self.index = faiss.IndexFlatIP(embed.shape[1])


if __name__ == "__main__":
    start = time.time()
    router = SemanticRouter(redis_host="vm.orannet.icu", redis_password="orange")
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )
    elapsed = time.time() - start
    print(f"初始化耗时:{elapsed:.3f}秒")

    for i in range(100):
        start = time.time()
        result = router("Hi, good morning")
        elapsed = time.time() - start
        print(f"路由结果:{result},耗时:{elapsed * 1000}毫秒")
    while True:
        query = input("输入提问，查看路由")
        if query is None or len(query) == 0:
            exit(0)
        start = time.time()
        result = router(query)
        elapsed = time.time() - start
        print(f"路由结果:{result},耗时:{elapsed * 1000}毫秒")
        time.sleep(0.5)
