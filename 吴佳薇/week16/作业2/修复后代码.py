import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss
import hashlib  # 添加缺失的导入


def _make_id(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,  # 过期时间
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
        if len(prompt) != len(response):
            raise ValueError("prompt and response must have same length")

        # embedding 标准化
        embedding = np.asarray(self.embedding_method(prompt), dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1])

        # 在 add 之前，读取当前索引大小（可用于校验）
        start_ntotal = self.index.ntotal
        self.index.add(embedding)
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    key_id = _make_id(q)
                    # 保存 answer（带前缀），并把 key_id 追加到列表尾（RPUSH）
                    pipe.setex(f"{self.name}:key:{key_id}", self.ttl, a)
                    pipe.rpush(f"{self.name}:list", key_id)
                return pipe.execute()
        except redis.RedisError:
            raise

    def call(self, prompt: str):
        if self.index is None or self.index.ntotal == 0:
            return None

        embedding = np.asarray(self.embedding_method(prompt), dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        k = min(100, max(1, self.index.ntotal))
        dis, ind = self.index.search(embedding, k=k)

        if dis.shape[0] == 0 or dis[0][0] > self.distance_threshold:
            return None

        # 取实际 faiss 中返回的 id（不是 enumerate 的位置）
        filtered_ids = [ind[0][i] for i, d in enumerate(dis[0]) if d < self.distance_threshold]

        # 从 redis list 中取出全部 key_id（按插入顺序，list 的索引应该与 faiss id 对齐）
        key_ids = self.redis.lrange(f"{self.name}:list", 0, -1)
        # key_ids 是 bytes，需要按 faiss id 索引
        filtered_key_ids = [key_ids[i].decode() for i in filtered_ids if i < len(key_ids)]

        keys = [f"{self.name}:key:{kid}" for kid in filtered_key_ids]
        if not keys:
            return None
        return self.redis.mget(keys)

    def clear_cache(self):
        # 修复1：正确的键名格式
        key_ids = self.redis.lrange(f"{self.name}:list", 0, -1)

        # 修复2：构建完整的键名列表
        keys_to_delete = []
        for key_id in key_ids:
            # key_id 是 bytes 类型，需要解码
            decoded_key_id = key_id.decode('utf-8')
            keys_to_delete.append(f"{self.name}:key:{decoded_key_id}")

        # 添加列表键本身
        keys_to_delete.append(f"{self.name}:list")

        # 修复3：删除所有键（如果列表不为空）
        if keys_to_delete:
            self.redis.delete(*keys_to_delete)

        # 修复4：安全删除索引文件
        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")
        self.index = None


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        # 修改：使用不同的向量，避免所有向量相同导致搜索问题
        # 使用简单的hash作为基础创建不同的向量
        return np.array([np.ones(768) * (hash(t) % 100) for t in text])


    # 修改：确保Redis服务正在运行
    embed_cache = SemanticCache(
        name="semantic_cache",  # 修正拼写错误
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    # 先尝试ping Redis，确保连接正常
    try:
        embed_cache.redis.ping()
        print("Redis连接成功")
    except redis.exceptions.ConnectionError:
        print("错误：无法连接到Redis服务器，请确保Redis已启动")
        exit(1)

    # 清理缓存
    embed_cache.clear_cache()
    print("缓存清理完成")

    # 存储第一个项目
    embed_cache.store(prompt="hello world", response="hello world1232")
    result1 = embed_cache.call(prompt="hello world")
    print(f"第一次查询结果: {result1}")

    # 存储第二个项目
    embed_cache.store(prompt="hello my name", response="nihao")
    result2 = embed_cache.call(prompt="hello world")
    print(f"第二次查询结果: {result2}")

    # 测试相似查询
    result3 = embed_cache.call(prompt="hello world!")
    print(f"相似查询结果: {result3}")
