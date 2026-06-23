import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss

class SemanticCache:
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

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        embedding = self.embedding_method(prompt)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1]) #IndexFlatL2 表示使用暴力搜索的平面索引，基于 L2 距离（欧几里得距离）

        self.index.add(embedding)
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    pipe.setex(self.name + "key:" + q, self.ttl, a) # 提问和回答存储在redis
                    pipe.lpush(self.name + "list", q) #所有的提问都存储在list里面，方便后续使用

                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: str):
        if self.index is None:
            return None

        # 新的提问进行编码
        embedding = self.embedding_method(prompt)

        # 向量数据库中进行检索
        dis, ind = self.index.search(embedding, k=100)
        if dis[0][0] > self.distance_threshold:
            return None

        # 过滤不满足距离的结果
        filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]

        pormpts = self.redis.lrange(self.name + "list", 0, -1)
        print("pormpts", pormpts)
        filtered_prompts = [pormpts[i] for i in filtered_ind]

        # 获取得到原始的提问 ，并在redis 找到对应的回答
        return self.redis.mget([self.name + "key:"+ q.decode() for q in filtered_prompts])

    def clear_cache(self):
        try:
            pormpts = self.redis.lrange(self.name + "list", 0, -1)
            if pormpts:  # 只有当列表不为空时才执行删除操作
                self.redis.delete(*(self.name + "key:" + p.decode() if isinstance(p, bytes) else p for p in pormpts))
            self.redis.delete(self.name + "list")
            if os.path.exists(f"{self.name}.index"):
                os.unlink(f"{self.name}.index")
            self.index = None
        except FileNotFoundError:
            self.index = None
        except Exception as e:
            print(f"Error clearing cache: {e}")

if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]

        return np.array([np.ones(768) for t in text])


    embed_cache = SemanticCache(
        name="semantic_ache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()

    embed_cache.store(prompt="hello world", response="hello world1232")
    print(embed_cache.call(prompt="hello world"))

    embed_cache.store(prompt="hello my bame", response="nihao")
    print(embed_cache.call(prompt="hello world"))
