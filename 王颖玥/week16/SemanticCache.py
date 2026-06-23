import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss

"""
实现语义缓存：基于文本的语义向量相似度，缓存「提问（prompt）- 回答（response）」对
"""


class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,  # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.1  # 向量相似度阈值（Faiss 用 L2 距离，值越小越相似，超过阈值则认为不匹配）
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

        if os.path.exists(f"{self.name}.index"):  # 检查是否存在Faiss索引文件
            self.index = faiss.read_index(f"{self.name}.index")   # 加载已有索引
        else:
            self.index = None  # 无索引文件则初始化索引为None

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        """
        将「提问 - 回答」对存储到 Faiss 索引（向量）+ Redis（文本）
        :param prompt: 提问文本，单个 / 列表
        :param response: 回答文本，单个 / 列表
        :return:
        """
        if isinstance(prompt, str):  # 统一格式：单个prompt转列表
            prompt = [prompt]
            response = [response]  # 同步转response为列表，保证一一对应

        embedding = self.embedding_method(prompt)  # 调用嵌入函数，将prompt转为向量
        if self.index is None:  # 若索引未初始化（首次存储）
            self.index = faiss.IndexFlatL2(embedding.shape[1])  # 创建L2距离的Flat索引（基础索引，适合小数据量）

        self.index.add(embedding)  # 将prompt的向量添加到Faiss索引
        faiss.write_index(self.index, f"{self.name}.index")  # 保存索引到文件（持久化）

        try:
            with self.redis.pipeline() as pipe:  # 创建Redis管道（批量执行，提升效率）
                for q, a in zip(prompt, response):  # 遍历prompt和response（一一对应）
                    # 存储prompt-response到Redis，Key格式：命名空间+key:+prompt文本
                    pipe.setex(self.name + "key:" + q, self.ttl, a)
                    # 将prompt添加到Redis的List中（Key：命名空间+list），方便后续批量查询
                    pipe.rpush(self.name + "list", q)

                return pipe.execute()  # 执行管道中所有命令，返回执行结果（[True, True, ...]）
        except:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: Union[str, List[str]]):
        """
        输入新 prompt，检索语义相似的历史 prompt，返回对应的 response
        :param prompt:
        :return:
        """
        if isinstance(prompt, str):  # 统一格式：单个prompt转列表
            prompt = [prompt]
            single = True
        else:
            single = False

        if self.index is None:  # 无Faiss索引（未存储过任何prompt）
            return [None] if single else [None] * len(prompt)

        # 新的提问进行编码
        embedding = self.embedding_method(prompt)  # 新prompt转为向量

        # 向量数据库中进行检索
        diss, inds = self.index.search(embedding, k=100)  # Faiss检索相似向量
        # dis：距离数组（shape=(1,100)），每个值是新向量与匹配向量的 L2 距离；
        # ind：索引位置数组（shape=(1,100)），每个值是匹配向量在 Faiss 索引中的位置（对应 Redis List 中的 prompt 位置）

        all_prompts = self.redis.lrange(self.name + "list", 0, -1)  # 取出Redis List中所有prompt
        results = []
        for dis, ind in zip(diss, inds):
            # 过滤不满足距离的结果 遍历 Top-100 的距离，保留 “距离小于阈值” 的向量索引位置
            filtered_ind = [int(i) for i, d in zip(ind, dis) if d < self.distance_threshold]
            if not filtered_ind:
                results.append(None)
                continue

            filtered_prompts = [all_prompts[i] for i in filtered_ind]  # 按索引位置匹配历史prompt
            keys = [self.name + "key:" + q.decode() for q in filtered_prompts]
            responses = self.redis.mget(keys)

            responses = [r for r in responses if r is not None]
            results.append(responses if responses else None)

        return results[0] if single else results

    # 清空 Redis 所有相关数据 + Faiss 索引文件，恢复初始状态
    def clear_cache(self):
        pormpts = self.redis.lrange(self.name + "list", 0, -1)  # 取出所有历史prompt

        if pormpts:
            delete_keys = [self.name + "key:" + q.decode() for q in pormpts]
            self.redis.delete(*delete_keys)

        self.redis.delete(self.name + "list")  # 删除存储prompt的List

        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")   # 删除Faiss索引文件

        self.index = None   # 重置索引为None


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]

        return np.array([np.ones(768) for t in text])


    embed_cache = SemanticCache(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()  # 清空历史缓存（测试前初始化）

    embed_cache.store(prompt="hello world", response="hello world1232")  # 存储第一个prompt-response
    print(embed_cache.call(prompt="hello world"))  # 查询，预期返回[b'hello world1232']

    embed_cache.store(prompt="hello my name", response="nihao")  # 存储第二个prompt-response
    print(embed_cache.call(prompt="hello world"))  # 查询，预期返回两个匹配结果（因为向量全1，距离为0）
    print(embed_cache.call(["hello world", "hello my name"]))
