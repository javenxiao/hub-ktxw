# 修复后的完整代码
import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any, Tuple
import faiss
import json
import time
import hashlib
import pickle
from threading import Lock


class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,  # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold: float = 0.1,
            max_cache_size: int = 10000  # 最大缓存数量
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False  # 保持字节类型，统一处理
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        self.max_cache_size = max_cache_size
        self.lock = Lock()  # 用于并发控制

        # 加载Faiss索引
        self.index_file = f"{self.name}.index"
        self.metadata_file = f"{self.name}_metadata.pkl"

        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = None
            self.metadata = {
                'id_to_key': {},  # Faiss ID -> Redis key
                'key_to_id': {},  # Redis key -> Faiss ID
                'next_id': 0
            }

    def _generate_key(self, prompt: str) -> str:
        """生成唯一的key"""
        return f"{self.name}:{hashlib.md5(prompt.encode()).hexdigest()}"

    def _get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        test_embedding = self.embedding_method(["test"])
        return test_embedding.shape[1]

    def _initialize_index(self, dim: int):
        """初始化Faiss索引"""
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = {
            'id_to_key': {},
            'key_to_id': {},
            'next_id': 0
        }

    def _cleanup_old_entries(self):
        """清理旧条目以控制缓存大小"""
        if len(self.metadata['id_to_key']) <= self.max_cache_size:
            return

        # 获取所有key的TTL，删除最旧的
        keys_to_delete = []
        with self.redis.pipeline() as pipe:
            for key in self.metadata['id_to_key'].values():
                pipe.ttl(key)
            ttls = pipe.execute()

        # 找到过期的或最旧的key
        current_time = time.time()
        for key, ttl in zip(self.metadata['id_to_key'].values(), ttls):
            if ttl < current_time * 0.1:  # TTL很小或已过期
                keys_to_delete.append(key)

        # 如果还需要删除更多，按时间排序
        if len(keys_to_delete) < (len(self.metadata['id_to_key']) - self.max_cache_size):
            # 这里简化处理：删除最后添加的
            keys_to_delete = list(self.metadata['id_to_key'].values())[
                             self.max_cache_size - len(self.metadata['id_to_key']):
                             ]

        for key in keys_to_delete:
            self._remove_entry(key)

    def _remove_entry(self, key: str):
        """删除单个条目"""
        if key in self.metadata['key_to_id']:
            faiss_id = self.metadata['key_to_id'][key]

            # 从Faiss索引中删除（标记删除）
            self.index.remove_ids(np.array([faiss_id], dtype=np.int64))

            # 从metadata中删除
            del self.metadata['id_to_key'][faiss_id]
            del self.metadata['key_to_id'][key]

            # 删除Redis中的数据
            self.redis.delete(key)

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]) -> bool:
        """存储prompt和response到缓存"""
        with self.lock:
            if isinstance(prompt, str):
                prompt = [prompt]
                response = [response]

            try:
                # 获取嵌入向量
                embeddings = self.embedding_method(prompt)

                # 初始化索引（如果需要）
                if self.index is None:
                    dim = embeddings.shape[1]
                    self._initialize_index(dim)

                # 批量存储
                with self.redis.pipeline() as pipe:
                    for p, r, emb in zip(prompt, response, embeddings):
                        # 生成唯一key
                        key = self._generate_key(p)

                        # 如果已存在，先删除旧条目
                        if key in self.metadata['key_to_id']:
                            self._remove_entry(key)

                        # 添加到Faiss索引
                        faiss_id = self.metadata['next_id']
                        self.index.add(emb.reshape(1, -1))

                        # 更新metadata
                        self.metadata['id_to_key'][faiss_id] = key
                        self.metadata['key_to_id'][key] = faiss_id
                        self.metadata['next_id'] += 1

                        # 存储到Redis
                        data = {
                            'prompt': p,
                            'response': r,
                            'embedding': emb.tolist(),
                            'timestamp': time.time()
                        }
                        pipe.setex(key, self.ttl, json.dumps(data))

                    # 执行Redis操作
                    pipe.execute()

                # 保存索引和metadata
                faiss.write_index(self.index, self.index_file)
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(self.metadata, f)

                # 清理旧条目
                self._cleanup_old_entries()

                return True

            except Exception as e:
                print(f"存储失败: {e}")
                import traceback
                traceback.print_exc()
                return False

    def call(self, prompt: str, k: int = 5) -> List[Tuple[str, float]]:
        """查找最相似的缓存条目"""
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            try:
                # 获取查询的嵌入向量
                query_embedding = self.embedding_method(prompt)

                # 搜索最相似的k个
                distances, indices = self.index.search(query_embedding, k)

                results = []
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    dist = distances[0][i]

                    # 跳过无效索引
                    if idx < 0 or idx >= self.index.ntotal:
                        continue

                    # 检查距离阈值
                    if dist > self.distance_threshold:
                        continue

                    # 获取对应的key
                    if idx in self.metadata['id_to_key']:
                        key = self.metadata['id_to_key'][idx]

                        # 从Redis获取数据
                        data_json = self.redis.get(key)
                        if data_json:
                            data = json.loads(data_json)
                            results.append((data['response'], float(dist)))

                # 按距离排序
                results.sort(key=lambda x: x[1])
                return results

            except Exception as e:
                print(f"检索失败: {e}")
                return []

    def get_exact(self, prompt: str) -> Optional[str]:
        """精确匹配获取缓存结果"""
        key = self._generate_key(prompt)
        data_json = self.redis.get(key)
        if data_json:
            data = json.loads(data_json)
            return data['response']
        return None

    def clear_cache(self) -> bool:
        """清空所有缓存"""
        with self.lock:
            try:
                # 删除所有相关的Redis键
                keys = self.redis.keys(f"{self.name}:*")
                if keys:
                    self.redis.delete(*keys)

                # 删除索引文件
                if os.path.exists(self.index_file):
                    os.unlink(self.index_file)

                if os.path.exists(self.metadata_file):
                    os.unlink(self.metadata_file)

                # 重置状态
                self.index = None
                self.metadata = {
                    'id_to_key': {},
                    'key_to_id': {},
                    'next_id': 0
                }

                return True

            except Exception as e:
                print(f"清空缓存失败: {e}")
                return False

    def stats(self) -> dict:
        """获取缓存统计信息"""
        with self.lock:
            try:
                total_keys = len(self.metadata['key_to_id'])
                index_size = self.index.ntotal if self.index else 0

                # 检查一致性
                consistent = total_keys == index_size

                return {
                    'name': self.name,
                    'total_entries': total_keys,
                    'index_size': index_size,
                    'consistent': consistent,
                    'max_size': self.max_cache_size,
                    'distance_threshold': self.distance_threshold
                }
            except:
                return {'error': '无法获取统计信息'}


# 测试代码
if __name__ == "__main__":
    # 模拟嵌入方法
    def get_embedding(texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # 简单哈希作为模拟嵌入
            hash_val = hashlib.md5(text.encode()).hexdigest()
            # 转换为768维向量
            np.random.seed(int(hash_val[:8], 16) % (2 ** 32))
            embedding = np.random.randn(768).astype(np.float32)
            embeddings.append(embedding)

        return np.array(embeddings)


    # 创建缓存
    cache = SemanticCache(
        name="semantic_cache_test",
        embedding_method=get_embedding,
        ttl=3600,  # 1小时
        distance_threshold=0.3,
        max_cache_size=100
    )

    # 清空缓存
    cache.clear_cache()

    # 测试存储
    print("存储测试数据...")
    cache.store("hello world", "hello world response")
    cache.store("hello there", "hello there response")
    cache.store("good morning", "good morning response")

    # 测试检索
    print("\n检索 'hello world':")
    results = cache.call("hello world")
    for response, distance in results:
        print(f"  - {response} (距离: {distance:.4f})")

    # 测试精确匹配
    print("\n精确匹配 'hello world':")
    exact = cache.get_exact("hello world")
    print(f"  结果: {exact}")

    # 测试统计
    print("\n缓存统计:")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试清理
    print("\n清空缓存...")
    cache.clear_cache()
    print("清空后统计:", cache.stats())
