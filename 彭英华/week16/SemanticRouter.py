import json
from typing import Optional, List, Union, Any, Dict,Callable,Annotated
import faiss
import numpy as np
import redis
from pydantic import BaseModel,Field
import os
from sentence_transformers import SentenceTransformer
class Route(BaseModel):
    name:str
    references:List[str]
    distance_threshold:Annotated[float, Field(strict=True, gt=0, le=2)] = 0.5

class SemanticRouter:
    def __init__(
            self,
            name:str,
            top_k:int,
            routes: List[Route],
            embedding_method: Callable[[Union[str, List[str]]], Any],
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            ttl: int = 3600 * 24,
    ):
        self.top_k = top_k
        self.name = name
        self.routes = routes
        self.embedding_method = embedding_method
        self.ttl = ttl
        self.redis = redis.Redis(host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=True)
        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None
        self._store_routes_to_redis()
    
    # 将Route存储到Redis中
    def _store_routes_to_redis(self):
        for route in self.routes:
            route_dict = {
                "name": route.name,
                "references": json.dumps(route.references),
                "distance_threshold": route.distance_threshold
            }
            route_json = json.dumps(route_dict)
            self.redis.rpush(f"{self.name}:routes", route_json)
            self.redis.expire(f"{self.name}:routes", self.ttl)
    
    # 从redis中加载所有Route
    def get_routes_from_redis(self) -> List[Route]:
        """从 Redis 加载所有 Route 对象"""
        routes_json_list = self.redis.lrange(f"{self.name}:routes", 0, -1)

        loaded_routes = []
        for route_json in routes_json_list:
            # 反序列化 JSON
            route_dict = json.loads(route_json)

            # 创建 Route 对象
            route = Route(
                name=route_dict["name"],
                references=json.loads(route_dict["references"]),
                distance_threshold=route_dict["distance_threshold"]
            )
            loaded_routes.append(route)

        return loaded_routes
    
    # 添加Route
    def add_route(self,routes:List[Route]):
        results = []
        loaded_routes = self.get_routes_from_redis()
        for route in routes:
            if route not in loaded_routes:
                route_dict = {
                    "name": route.name,
                    "references": json.dumps(route.references),
                    "distance_threshold": route.distance_threshold
                }
                route_json = json.dumps(route_dict)
                self.redis.rpush(f"{self.name}:routes",route_json)
                self.redis.expire(f"{self.name}:routes", self.ttl)
                for reference in route.references:
                    duplicate = self.redis.get(self.name + "reference:" + reference)
                    if duplicate == None:
                        references_vectors = self.embedding_method([reference])
                        if self.index is None:
                            self.index = faiss.IndexFlatL2(references_vectors.shape[1])
                        self.index.add(references_vectors)
                        faiss.write_index(self.index, f"{self.name}.index")
                        try:
                            with self.redis.pipeline() as pipe:
                                pipe.setex(self.name + "reference:" + reference, self.ttl,json.dumps({"name":route.name,"distance_threshold":route.distance_threshold}))
                                pipe.rpush(self.name + "references", reference)

                                results.append(pipe.execute())
                        except:
                            import traceback
                            traceback.print_exc()
                            return -1
                    else:
                        continue
        return results
    
     # 意图匹配
    def route(self, question: str):
       # 如果redis中缓存了结果，直接返回，不用进行相似度检索
        result_json = self.redis.get(self.name + "reference:" + question)
        if result_json!=None:
            result_dict = json.loads(result_json)
            return {"name":result_dict['name'],"distance":result_dict['distance_threshold']}

      # 相似度检索
        if self.index is None:
            return {"name":None,"distance":None}
        self.routes = self.get_routes_from_redis()
        distances_threshold = max(route.distance_threshold for route in self.routes)
        question_vector = self.embedding_method(question)
        dis, ind = self.index.search(question_vector, k=30)
        if dis[0][0] > distances_threshold:
            return {"name":None,"distance":None}
        filtered_ind = [ind[0][i] for i, d in enumerate(dis[0]) if d < distances_threshold]
        filtered_dis = [dis[0][i] for i,d in enumerate(dis[0]) if d<distances_threshold]
        references = self.redis.lrange(self.name + "references",0,-1)
        filtered_references = [references[i] for i in filtered_ind]
        names = self.redis.mget([self.name + "reference:" + reference for reference in filtered_references])
        results = []
        for name,dis in zip(names,filtered_dis):
            results.append({"name":json.loads(name)['name'],"distance":dis})
        return results[:self.top_k]

    def clear_cache(self):
        pormpts = self.redis.lrange(self.name + "references", 0, -1)
        for i in pormpts:
            self.redis.delete(self.name + "reference:" + i)
        self.redis.delete(self.name + "references")
        self.redis.delete(f"{self.name}:routes")
        os.unlink(f"{self.name}.index")
        self.index = None





if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text,str):
            text = [text]
        model = SentenceTransformer("../models/Qwen3-Embedding",trust_remote_code=True)
        return np.array([model.encode_query(t) for t in text])
    routes = [
        Route(
            name="greeting",  # 类别的名字
            references=["hello", "hi"],  # 类别参考例子
            distance_threshold=0.3,
        ),
        Route(
            name="farewell",  # 类别的名字
            references=["bye", "goodbye"],  # 类别参考例子
            distance_threshold=0.3,
        ),
        Route(name="machine learning",
              references = ["MLP","deep learning"],
              distance_threshold = 0.4),
    ]
    routes_1 = [
        Route(
            name="greeting",  # 类别的名字
            references=["hello", "hi"],  # 类别参考例子
            distance_threshold=0.3,
        ),
        Route(
            name="farewell",  # 类别的名字
            references=["bye", "goodbye"],  # 类别参考例子
            distance_threshold=0.3,
        )
    ]
    router = SemanticRouter(name="router",top_k=1,routes=routes_1,embedding_method=get_embedding)
    # router.clear_cache()
    router.add_route(routes)
    print(router.route("hello"))
