import numpy as np
import redis
import json
import hashlib
from typing import Optional, List, Union, Any, Dict, Tuple
from collections import defaultdict


class Route:
    """路由类，表示一个意图类别"""
    
    def __init__(
        self,
        name: str,
        references: List[str],
        metadata: Optional[Dict] = None,
        distance_threshold: float = 0.3
    ):
        self.name = name
        self.references = references
        self.metadata = metadata or {}
        self.distance_threshold = distance_threshold
        self.embeddings = None  # 存储参考文本的嵌入向量
        
    def set_embeddings(self, embeddings: np.ndarray):
        """设置参考文本的嵌入向量"""
        if len(embeddings) != len(self.references):
            raise ValueError(f"嵌入向量数量({len(embeddings)})与参考文本数量({len(self.references)})不匹配")
        self.embeddings = embeddings
    
    def __repr__(self):
        return f"Route(name='{self.name}', references={len(self.references)} texts, threshold={self.distance_threshold})"


class SemanticRouter:
    def __init__(
        self,
        name: str,
        routes: Optional[List[Route]] = None,
        embedding_method: Optional[callable] = None,
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        use_cache: bool = True,
        cache_threshold: float = 0.1
    ):
        self.name = name
        self.routes = routes or []
        self.embedding_method = embedding_method
        self.use_cache = use_cache
        self.cache_threshold = cache_threshold
        
        # Redis连接
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True
        )
        self.ttl = ttl
        
        # 缓存相关键名
        self.cache_key_prefix = f"router_cache:{self.name}:"
        
        # 初始化时计算所有路由的嵌入向量
        if self.embedding_method and self.routes:
            self._compute_route_embeddings()
    
    def _compute_route_embeddings(self):
        """计算所有路由参考文本的嵌入向量"""
        for route in self.routes:
            if route.references:
                embeddings = self.embedding_method(route.references)
                route.set_embeddings(embeddings)
    
    def _get_cache_key(self, question: str) -> str:
        """生成缓存键"""
        # 使用哈希确保键名不会太长
        question_hash = hashlib.md5(question.encode()).hexdigest()
        return f"{self.cache_key_prefix}{question_hash}"
    
    def _check_cache(self, question: str) -> Optional[Dict]:
        """检查缓存中是否有结果"""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(question)
        cached_data = self.redis.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        return None
    
    def _store_cache(self, question: str, result: Dict):
        """存储结果到缓存"""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(question)
        self.redis.setex(
            cache_key,
            self.ttl,
            json.dumps(result)
        )
    
    def _compute_similarity(self, query_embedding: np.ndarray, route: Route) -> Tuple[float, float, int]:
        """
        计算查询与路由的相似度
        返回: (最小距离, 平均距离, 匹配的参考文本索引)
        """
        if route.embeddings is None or len(route.embeddings) == 0:
            return float('inf'), float('inf'), -1
        
        # 计算与每个参考文本的距离
        distances = []
        for ref_embedding in route.embeddings:
            # 使用余弦距离
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # 归一化向量
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            ref_norm = ref_embedding / np.linalg.norm(ref_embedding)
            
            # 计算余弦相似度
            similarity = np.dot(query_norm, ref_norm.T)[0][0]
            distance = 1 - similarity  # 余弦距离
            distances.append(distance)
        
        # 找到最小距离
        if distances:
            min_distance = min(distances)
            avg_distance = sum(distances) / len(distances)
            min_index = distances.index(min_distance)
            return min_distance, avg_distance, min_index
        
        return float('inf'), float('inf'), -1
    
    def add_route(self, route: Route):
        """添加路由"""
        self.routes.append(route)
        
        # 如果提供了嵌入方法，计算新路由的嵌入向量
        if self.embedding_method and route.references:
            embeddings = self.embedding_method(route.references)
            route.set_embeddings(embeddings)
    
    def add_routes(self, routes: List[Route]):
        """批量添加路由"""
        for route in routes:
            self.add_route(route)
    
    def route(self, question: str, return_details: bool = False) -> Union[str, Dict, None]:
        """
        对问题进行路由
        
        Args:
            question: 输入问题
            return_details: 是否返回详细信息
            
        Returns:
            如果return_details为False: 返回路由名称或None
            如果return_details为True: 返回包含详细信息的字典
        """
        # 检查缓存
        cached_result = self._check_cache(question)
        if cached_result:
            if not return_details:
                return cached_result.get("route_name")
            return cached_result
        
        # 如果没有嵌入方法或没有路由，返回None
        if not self.embedding_method or not self.routes:
            return None
        
        # 计算问题的嵌入向量
        query_embedding = self.embedding_method(question)
        
        # 初始化最佳匹配
        best_route = None
        best_distance = float('inf')
        best_details = {
            "route_name": None,
            "distance": float('inf'),
            "avg_distance": float('inf'),
            "matched_reference": None,
            "matched_reference_index": -1,
            "threshold": float('inf'),
            "is_match": False,
            "metadata": {}
        }
        
        # 遍历所有路由，找到最佳匹配
        for route in self.routes:
            min_distance, avg_distance, ref_index = self._compute_similarity(query_embedding, route)
            
            # 检查是否满足距离阈值
            if min_distance < route.distance_threshold and min_distance < best_distance:
                best_distance = min_distance
                best_route = route
                best_details.update({
                    "route_name": route.name,
                    "distance": min_distance,
                    "avg_distance": avg_distance,
                    "matched_reference": route.references[ref_index] if ref_index >= 0 else None,
                    "matched_reference_index": ref_index,
                    "threshold": route.distance_threshold,
                    "is_match": True,
                    "metadata": route.metadata.copy()
                })
        
        # 如果没有匹配的路由，检查是否有默认路由
        if best_route is None:
            # 查找名为"default"的路由
            for route in self.routes:
                if route.name == "default":
                    best_route = route
                    best_details.update({
                        "route_name": "default",
                        "is_match": True,
                        "metadata": route.metadata.copy()
                    })
                    break
        
        # 如果有缓存，并且匹配成功，存储到缓存
        if best_route and self.use_cache and best_details["is_match"]:
            # 只有在距离小于缓存阈值时才缓存
            if best_details["distance"] < self.cache_threshold:
                cache_data = best_details.copy()
                # 简化缓存数据，减少存储空间
                cache_data = {
                    "route_name": best_details["route_name"],
                    "distance": float(best_details["distance"]),
                    "is_match": True,
                    "cached_at": int(np.datetime64('now').astype(int) / 1e9)  # Unix时间戳
                }
                self._store_cache(question, cache_data)
        
        if not return_details:
            return best_details["route_name"] if best_details["is_match"] else None
        
        return best_details
    
    def batch_route(self, questions: List[str], return_details: bool = False) -> List:
        """批量路由"""
        results = []
        for question in questions:
            result = self.route(question, return_details)
            results.append(result)
        return results
    
    def get_route_stats(self) -> Dict:
        """获取路由统计信息"""
        stats = {
            "total_routes": len(self.routes),
            "routes": {},
            "total_references": 0
        }
        
        for route in self.routes:
            route_stats = {
                "name": route.name,
                "reference_count": len(route.references),
                "threshold": route.distance_threshold,
                "has_embeddings": route.embeddings is not None
            }
            stats["routes"][route.name] = route_stats
            stats["total_references"] += len(route.references)
        
        return stats
    
    def clear_cache(self):
        """清空当前路由器的缓存"""
        pattern = f"{self.cache_key_prefix}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
        return len(keys)
    
    def update_route_threshold(self, route_name: str, new_threshold: float):
        """更新路由的距离阈值"""
        for route in self.routes:
            if route.name == route_name:
                route.distance_threshold = new_threshold
                return True
        return False


# 测试代码
if __name__ == "__main__":
    # 模拟嵌入方法
    def mock_embedding_method(texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # 生成确定性嵌入向量用于测试
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(384).astype(np.float32)
            # 归一化
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]
    
    # 创建路由
    routes = [
        Route(
            name="greeting",
            references=["hello", "hi", "good morning", "good afternoon"],
            metadata={"type": "greeting", "priority": 1},
            distance_threshold=0.3
        ),
        Route(
            name="farewell",
            references=["bye", "goodbye", "see you", "take care"],
            metadata={"type": "farewell", "priority": 1},
            distance_threshold=0.3
        ),
        Route(
            name="refund",
            references=["how to refund", "return policy", "退货", "退款"],
            metadata={"type": "customer_service", "priority": 2},
            distance_threshold=0.25
        ),
        Route(
            name="shipping",
            references=["shipping time", "delivery", "物流", "配送"],
            metadata={"type": "customer_service", "priority": 2},
            distance_threshold=0.25
        ),
        Route(
            name="default",
            references=["other"],
            metadata={"type": "default", "priority": 0},
            distance_threshold=0.5
        )
    ]
    
    # 创建语义路由器
    router = SemanticRouter(
        name="customer-service-router",
        routes=routes,
        embedding_method=mock_embedding_method,
        ttl=3600,  # 1小时缓存
        use_cache=True,
        cache_threshold=0.1
    )
    
    print("路由器统计信息:")
    stats = router.get_route_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    print("\n测试路由功能:")
    
    # 测试用例
    test_cases = [
        "Hi, good morning",
        "Hi, good morning",  # 重复测试，应该从缓存获取
        "how to get a refund",
        "what's the return policy",
        "bye bye",
        "see you later",
        "shipping time for my order",
        "delivery status",
        "天气怎么样",  # 不在任何路由中，应该匹配到default或返回None
        "hello there"
    ]
    
    for i, question in enumerate(test_cases):
        print(f"\n{i+1}. 问题: '{question}'")
        
        # 第一次路由（可能从缓存获取）
        result = router.route(question, return_details=True)
        
        if result and result["is_match"]:
            print(f"   路由: {result['route_name']}")
            print(f"   距离: {result['distance']:.4f} (阈值: {result['threshold']})")
            if result.get("matched_reference"):
                print(f"   匹配的参考: {result['matched_reference']}")
            if result.get("metadata"):
                print(f"   元数据: {result['metadata']}")
            
            # 检查缓存状态
            cache_key = router._get_cache_key(question)
            cached = router.redis.get(cache_key)
            if cached:
                print(f"   ✓ 已缓存")
            else:
                print(f"   - 未缓存（距离 {result['distance']:.4f} > 缓存阈值 {router.cache_threshold}）")
        else:
            print(f"   ✗ 未匹配到任何路由")
    
    print("\n批量路由测试:")
    batch_results = router.batch_route(["hello", "how to refund", "unknown question"], return_details=False)
    for q, r in zip(["hello", "how to refund", "unknown question"], batch_results):
        print(f"  '{q}' -> {r}")
    
    print(f"\n清空缓存，删除了 {router.clear_cache()} 个缓存项")
    
    print("\n更新路由阈值测试:")
    router.update_route_threshold("greeting", 0.2)
    updated_stats = router.get_route_stats()
    print(f"  更新后greeting路由阈值: {updated_stats['routes']['greeting']['threshold']}")