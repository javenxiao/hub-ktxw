import hashlib
import json
from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any

import faiss
import numpy as np
import redis


@dataclass
class Route:
    """路由定义"""
    name: str
    references: List[str]
    metadata: Dict[str, Any]
    distance_threshold: float


class RouteResult:
    """路由结果"""

    def __init__(self, route: Optional[Route] = None, distance: float = float('inf'), cached: bool = False):
        self.route = route
        self.distance = distance
        self.cached = cached
        self.name = route.name if route else None
        self.metadata = route.metadata if route else {}


class SemanticRouter:
    def __init__(
            self,
            name: str,
            routes: List[Route],
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: Optional[str] = None,
            ttl: int = 3600 * 24,
            embedding_method=None,
            default_embedding_dim: int = 384  # 默认维度，可以适配常见的embedding模型
    ):
        self.name = name
        self.routes = routes
        self.ttl = ttl
        self.embedding_method = embedding_method
        self.embedding_dim = default_embedding_dim

        # 初始化Redis连接
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )

        # 初始化向量索引
        self._init_indices()

        # 初始化缓存键前缀
        self.cache_prefix = f"router:{name}:cache"

    def _init_indices(self):
        """初始化每个route的向量索引"""
        self.indices = {}
        self.route_embeddings = {}

        for route in self.routes:
            # 为每个route创建向量索引
            if route.references:
                embeddings = self._embed_texts(route.references)
                self.route_embeddings[route.name] = embeddings

                # 创建FAISS索引
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                self.indices[route.name] = index
            else:
                self.route_embeddings[route.name] = None
                self.indices[route.name] = None

    def _embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """文本向量化"""
        if isinstance(texts, str):
            texts = [texts]

        if self.embedding_method:
            # 使用指定的embedding方法
            return self.embedding_method(texts)
        else:
            # 使用简单的随机embedding（实际使用时应该替换为真实的embedding模型）
            embeddings = []
            for text in texts:
                # 使用hash作为种子，确保相同文本得到相同embedding
                seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                np.random.seed(seed)
                embeddings.append(np.random.randn(self.embedding_dim))

            return np.array(embeddings).astype(np.float32)

    def _get_cache_key(self, text: str) -> str:
        """获取缓存键"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.cache_prefix}:{text_hash}"

    def _get_cached_result(self, text: str) -> Optional[RouteResult]:
        """获取缓存结果"""
        cache_key = self._get_cache_key(text)
        cached_data = self.redis.get(cache_key)

        if cached_data:
            data = json.loads(cached_data)
            # 查找对应的route
            route_name = data.get("route_name")
            route = next((r for r in self.routes if r.name == route_name), None)

            return RouteResult(
                route=route,
                distance=data.get("distance", float('inf')),
                cached=True
            )
        return None

    def _set_cached_result(self, text: str, result: RouteResult):
        """设置缓存结果"""
        if result.route:
            cache_key = self._get_cache_key(text)
            cache_data = {
                "route_name": result.route.name,
                "distance": float(result.distance),
                "metadata": result.route.metadata
            }
            self.redis.setex(cache_key, self.ttl, json.dumps(cache_data))

    def _calculate_distance(self, query_embedding: np.ndarray, route_name: str) -> float:
        """计算查询与route的最小距离"""
        if route_name not in self.indices or self.indices[route_name] is None:
            return float('inf')

        index = self.indices[route_name]
        # 搜索最近的一个向量
        distances, _ = index.search(query_embedding, k=1)

        if distances.size > 0:
            # 使用L2距离
            return float(np.sqrt(distances[0][0]))
        return float('inf')

    def call(self, text: str, use_cache: bool = True) -> RouteResult:
        """
        对文本进行意图路由

        Args:
            text: 输入文本
            use_cache: 是否使用缓存

        Returns:
            RouteResult: 路由结果
        """
        # 1. 检查缓存
        if use_cache:
            cached_result = self._get_cached_result(text)
            if cached_result:
                return cached_result

        # 2. 向量化查询文本
        query_embedding = self._embed_texts([text])

        # 3. 计算与每个route的距离
        best_result = RouteResult()

        for route in self.routes:
            distance = self._calculate_distance(query_embedding, route.name)

            # 如果距离小于阈值，并且比当前最佳结果更近
            if distance <= route.distance_threshold and distance < best_result.distance:
                best_result = RouteResult(route=route, distance=distance)

        # 4. 缓存结果
        self._set_cached_result(text, best_result)

        return best_result

    def batch_predict(self, texts: List[str], use_cache: bool = True) -> List[RouteResult]:
        """批量预测"""
        results = []

        for text in texts:
            result = self(text, use_cache)
            results.append(result)

        return results

    def add_route(self, route: Route):
        """添加新的路由"""
        self.routes.append(route)

        # 重新初始化索引
        if route.references:
            embeddings = self._embed_texts(route.references)
            self.route_embeddings[route.name] = embeddings

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            self.indices[route.name] = index
        else:
            self.route_embeddings[route.name] = None
            self.indices[route.name] = None

    def remove_route(self, route_name: str):
        """移除路由"""
        self.routes = [r for r in self.routes if r.name != route_name]

        if route_name in self.indices:
            del self.indices[route_name]
        if route_name in self.route_embeddings:
            del self.route_embeddings[route_name]

    def clear_cache(self):
        """清除所有缓存"""
        pattern = f"{self.cache_prefix}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pattern = f"{self.cache_prefix}:*"
        keys = self.redis.keys(pattern)

        stats = {
            "total_cached": len(keys),
            "routes_distribution": {}
        }

        # 统计每个route的缓存数量
        for key in keys:
            data = self.redis.get(key)
            if data:
                route_data = json.loads(data)
                route_name = route_data.get("route_name", "unknown")
                stats["routes_distribution"][route_name] = stats["routes_distribution"].get(route_name, 0) + 1

        return stats


if __name__ == "__main__":
    # 定义路由
    routes = [
        Route(
            name="greeting",
            references=["hello", "hi", "good morning", "hey there"],
            metadata={"type": "greeting", "priority": 1},
            distance_threshold=0.3,
        ),
        Route(
            name="farewell",
            references=["bye", "goodbye", "see you", "take care"],
            metadata={"type": "farewell", "priority": 1},
            distance_threshold=0.3,
        ),
        Route(
            name="question",
            references=["what is", "how to", "can you", "why is"],
            metadata={"type": "question", "priority": 2},
            distance_threshold=0.4,
        ),
    ]

    # 创建路由器
    router = SemanticRouter(
        name="topic-router",
        routes=routes,
        redis_url="localhost",
    )

    # 测试
    test_texts = [
        "Hi, good morning",
        "Hi, good morning",  # 第二次调用应该从缓存获取
        "Goodbye, see you tomorrow",
        "What is the weather like?",
        "Hello, how are you?",
        "Can you help me with this?",
        "This is not related to any route"
    ]

    print("第一次批量预测：")
    results = router.batch_predict(test_texts)
    for text, result in zip(test_texts, results):
        print(f"  '{text}' -> {result}")

    print("\n第二次调用（应该使用缓存）：")
    for text in test_texts[:3]:
        result = router(text)
        print(f"  '{text}' -> {result}")

    print("\n缓存统计：")
    stats = router.get_cache_stats()
    print(f"  总缓存数: {stats['total_cached']}")
    for route_name, count in stats['routes_distribution'].items():
        print(f"  {route_name}: {count}")

    # 清理缓存
    router.clear_cache()
    print(f"\n清理后缓存数: {len(router.redis.keys('router:topic-router:cache:*'))}")
