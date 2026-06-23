import numpy as np
import re
from typing import Dict, Callable
from sentence_transformers import SentenceTransformer
from functools import wraps


class ToolRegistry:
    def __init__(self, embedding_model_name=r"F:\ai\tools\BAAI\bge-small-zh-v1.5"):
        """初始化工具注册表"""
        self.tools = {}
        self.tool_embeddings = {}
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def register_tool(self, func: Callable) -> Callable:
        """装饰器：自动注册工具到注册表"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # 提取工具信息
        tool_name = func.__name__
        description = func.__doc__ or ""

        # 简化描述，去除详细说明
        simple_desc = description.split('\n')[0] if description else tool_name

        # 存储工具信息
        self.tools[tool_name] = {
            'name': tool_name,
            'description': simple_desc,
            'full_description': description,
            'function': wrapper,
            'docstring': description
        }

        # 为工具生成嵌入向量
        tool_text = f"{tool_name}: {simple_desc}"
        embedding = self.embedding_model.encode(tool_text, normalize_embeddings=True)
        self.tool_embeddings[tool_name] = embedding

        return wrapper

    def find_most_similar_tool(self, query: str) -> tuple:
        """使用RAG找到最相似的工具"""
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)

        best_match = None
        best_similarity = -1

        for name, tool_embedding in self.tool_embeddings.items():
            similarity = np.dot(query_embedding, tool_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = self.tools[name]

        return best_match, best_similarity

    def get_tool(self, tool_name: str) -> Dict:
        """获取特定工具"""
        return self.tools.get(tool_name)

    def list_tools(self) -> list:
        """列出所有工具"""
        return list(self.tools.keys())

    def extract_parameters_from_docstring(self, docstring: str) -> list:
        """从docstring中提取参数名"""
        param_pattern = r': param (\w+)：'
        return re.findall(param_pattern, docstring)
