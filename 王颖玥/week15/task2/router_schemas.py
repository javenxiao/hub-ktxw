from pydantic import BaseModel, Field
from typing import Union, List, Any, Tuple, Dict
from fastapi import FastAPI, File, UploadFile, Form
from typing_extensions import Annotated

"""
主要用于定义API接口的请求数据格式和响应数据格式
"""

# 语义嵌入请求
"""
示例：
{
  "text": ["这是一个测试句子", "另一个需要转换的文本"],
  "token": "user_auth_token_123",
  "model": "bge-small-zh-v1.5"
}
"""
class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]  # 输入文本：可以是单个字符串，或字符串列表（批量处理）
    token: str  # 用户认证令牌（类似“钥匙”，验证用户是否有权限调用接口）
    model: str  # 指定使用的嵌入模型（比如"bge-small-zh-v1.5"是一个中文嵌入模型）


# 语义嵌入响应
class EmbeddingResponse(BaseModel):
    request_id: str = Field(description="请求ID")  # 用Field添加描述，说明字段含义
    vector: List[List[float]] = Field(description="文本对应的向量表示")  # 向量列表（每个文本对应一个子列表）
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


# 重排序请求
"""
示例：
{
  "text_pair": [
    ("什么是RAG？", "RAG是检索增强生成的缩写"),
    ("什么是RAG？", "机器学习是人工智能的分支")
  ],
  "token": "user_auth_token_123",
  "model": "bge-reranker-base"
}
"""
class RerankRequest(BaseModel):
    text_pair: List[Tuple[str, str]]  # 文本对列表：每个元素是(查询文本, 候选文档文本)的元组
    token: str
    model: str  # 指定使用的重排序模型

# 重排序响应
class RerankResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    vector: List[float]  # 相关性分数列表：每个元素对应text_pair中一个文本对的分数（越高越相关）
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")


# RAG聊天请求
"""
示例：
{
  "knowledge_id": 1001,
  "message": [
    {"role": "user", "content": "文档中提到的Python版本要求是什么？"}
  ]
}
"""
class Message(BaseModel):
    role: str
    content: str

class RAGRequest(BaseModel):
    message: List[Message]  # 聊天消息列表：每个元素是包含"role"（角色）和"content"（内容）的字典

# RAG聊天响应
class RAGResponse(BaseModel):
    request_id: str = Field(description="请求ID")
    message: List[Message]  # 回答消息列表：每个元素是包含"role"和"content"的字典（通常"role"为"assistant"）
    response_code: int = Field(description="响应代码，用于表示成功或错误状态")
    response_msg: str = Field(description="响应信息，详细描述响应状态或错误信息")
    process_status: str = Field(description="处理状态，例如 'completed'、'pending' 或 'failed'")
    processing_time: float = Field(description="处理请求的耗时（秒）")
