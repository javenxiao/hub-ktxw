import os
import json
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any, Dict
from dataclasses import dataclass
import faiss
from urllib.parse import urlparse


@dataclass
class Route:
    """路由定义类"""
    name: str  # 路由名称
    references: List[str]  # 参考文本列表，用于定义该路由的语义特征
    metadata: Optional[Dict[str, Any]] = None  # 路由元数据
    distance_threshold: float = 0.3  # 距离阈值，小于此值才匹配


@dataclass
class RouteMatch:
    """路由匹配结果类"""
    name: str  # 匹配的路由名称
    distance: float  # 匹配距离
    metadata: Optional[Dict[str, Any]] = None  # 路由元数据


class SemanticRouter:
    """语义路由类，用于根据查询文本的语义相似度匹配到最近的路由"""
    
    def __init__(
            self,
            name: str,
            routes: List[Route],
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int=3600*24, # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None
    ):
        self.name = name
        self.routes = routes
        self.embedding_method = embedding_method
        self.ttl = ttl
        
        # 连接 Redis
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False  # 保持 bytes 格式以便与 FAISS 索引对应
        )
        
        # 初始化或加载 FAISS 索引
        index_path = f"{self.name}.index"
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = None
        
        # 构建路由索引
        self._build_routes()
    
    def _build_routes(self):
        """构建路由索引，将所有路由的 references 添加到 FAISS 索引中"""
        if not self.routes:
            return
        
        # 收集所有 references 文本及其对应的路由信息
        all_references = []
        route_info = []  # 存储每个 reference 对应的路由信息
        
        for route in self.routes:
            for ref in route.references:
                all_references.append(ref)
                route_info.append({
                    "route_name": route.name,
                    "distance_threshold": route.distance_threshold,
                    "metadata": route.metadata
                })
        
        if not all_references:
            return
        
        # 计算所有 references 的 embedding
        embeddings = self.embedding_method(all_references)
        
        # 确保 embeddings 是 numpy 数组
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # 如果是单个向量，需要 reshape
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # 创建或扩展 FAISS 索引
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        # 添加向量到索引
        self.index.add(embeddings.astype(np.float32))
        
        # 保存索引到磁盘
        faiss.write_index(self.index, f"{self.name}.index")
        
        # 将路由信息存储到 Redis
        try:
            with self.redis.pipeline() as pipe:
                # 存储每个 reference 对应的路由信息
                for i, (ref, info) in enumerate(zip(all_references, route_info)):
                    key = f"{self.name}:ref:{i}"
                    pipe.setex(key, self.ttl, json.dumps(info))
                    # 存储 reference 文本，用于后续查找
                    pipe.setex(f"{self.name}:text:{i}", self.ttl, ref)
                
                # 存储路由信息列表的索引范围
                pipe.setex(f"{self.name}:route_info", self.ttl, json.dumps({
                    "total_refs": len(all_references),
                    "route_names": [route.name for route in self.routes]
                }))
                
                pipe.execute()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"存储路由信息到 Redis 失败: {e}")
    
    def __call__(self, query: str) -> Optional[RouteMatch]:
        """
        对查询文本进行路由匹配
        
        Args:
            query: 查询文本
            
        Returns:
            RouteMatch 对象，如果找到匹配的路由；否则返回 None
        """
        # 检查缓存
        cached_result = self._get_cached_result(query)
        if cached_result is not None:
            return cached_result
        
        if self.index is None or self.index.ntotal == 0:
            return None
        
        # 计算查询文本的 embedding
        query_embedding = self.embedding_method(query)
        
        # 确保是 numpy 数组
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # 如果是单个向量，需要 reshape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype(np.float32)
        
        # 在 FAISS 索引中搜索最近的向量
        k = min(10, self.index.ntotal)  # 搜索 top-k，最多 10 个
        distances, indices = self.index.search(query_embedding, k)
        
        if len(distances[0]) == 0:
            return None
        
        # 获取所有候选路由及其距离
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            # 从 Redis 获取路由信息
            route_info_key = f"{self.name}:ref:{idx}"
            route_info_str = self.redis.get(route_info_key)
            
            if route_info_str:
                route_info = json.loads(route_info_str)
                candidates.append({
                    "route_name": route_info["route_name"],
                    "distance": float(dist),
                    "threshold": route_info["distance_threshold"],
                    "metadata": route_info.get("metadata")
                })
        
        if not candidates:
            return None
        
        # 找到距离最近且满足阈值要求的路由
        # 按距离排序
        candidates.sort(key=lambda x: x["distance"])
        
        # 找到第一个满足阈值要求的路由
        for candidate in candidates:
            if candidate["distance"] <= candidate["threshold"]:
                result = RouteMatch(
                    name=candidate["route_name"],
                    distance=candidate["distance"],
                    metadata=candidate["metadata"]
                )
                # 缓存结果
                self._cache_result(query, result)
                return result
        
        # 如果没有满足阈值要求的，返回 None
        return None
    
    def _get_cached_result(self, query: str) -> Optional[RouteMatch]:
        """从缓存中获取查询结果"""
        try:
            # 使用查询文本的哈希作为缓存 key
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cache_key = f"{self.name}:cache:{query_hash}"
            
            cached_data = self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return RouteMatch(
                    name=data["name"],
                    distance=data["distance"],
                    metadata=data.get("metadata")
                )
        except Exception as e:
            # 缓存获取失败不影响主流程
            pass
        
        return None
    
    def _cache_result(self, query: str, result: RouteMatch):
        """缓存查询结果"""
        try:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cache_key = f"{self.name}:cache:{query_hash}"
            
            cache_data = {
                "name": result.name,
                "distance": result.distance,
                "metadata": result.metadata
            }
            
            self.redis.setex(cache_key, self.ttl, json.dumps(cache_data))
        except Exception as e:
            # 缓存失败不影响主流程
            pass
    
    def clear_cache(self):
        """清除路由缓存和索引"""
        try:
            # 删除 Redis 中的路由信息
            route_info_key = f"{self.name}:route_info"
            route_info_str = self.redis.get(route_info_key)
            
            if route_info_str:
                route_info = json.loads(route_info_str)
                total_refs = route_info.get("total_refs", 0)
                
                # 删除所有 reference 信息
                keys_to_delete = [f"{self.name}:route_info"]
                for i in range(total_refs):
                    keys_to_delete.append(f"{self.name}:ref:{i}")
                    keys_to_delete.append(f"{self.name}:text:{i}")
                
                # 删除所有缓存结果（使用模式匹配）
                pattern = f"{self.name}:cache:*"
                for key in self.redis.scan_iter(match=pattern):
                    keys_to_delete.append(key)
                
                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
            
            # 删除 FAISS 索引文件
            index_path = f"{self.name}.index"
            if os.path.exists(index_path):
                os.unlink(index_path)
            
            self.index = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"清除缓存失败: {e}")


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]

        return np.array([np.ones(768) for t in text])
    
    # 定义路由
    routes = [
        Route(
            name="greeting",
            references=["hello", "hi"],
            metadata={"type": "greeting"},
            distance_threshold=0.3,
        ),
        Route(
            name="farewell",
            references=["bye", "goodbye"],
            metadata={"type": "farewell"},
            distance_threshold=0.3,
        ),
    ]
    
    # 创建语义路由器
    router = SemanticRouter(
        name="topic-router",
        routes=routes,
        embedding_method=get_embedding,
        redis_url="localhost",
    )
    
    # 测试路由匹配
    result = router("Hi, good morning")
    print(f"查询: 'Hi, good morning'")
    print(f"匹配结果: {result}")
    
    # 再次查询相同内容，应该从缓存获取
    result2 = router("Hi, good morning")
    print(f"缓存查询结果: {result2}")
