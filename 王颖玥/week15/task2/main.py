import yaml   # type: ignore
with open("config.yaml", "r") as f:  # 从配置文件读取服务参数（如端口号）
    config = yaml.safe_load(f)

import time
import numpy as np
import uuid   # 生成唯一的“请求ID”（方便追踪每个用户请求）

import uvicorn  # 运行FastAPI服务的服务器
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks  # FastAPI核心组件

from router_schemas import (
    EmbeddingRequest, EmbeddingResponse,  
    RerankRequest, RerankResponse,        
    RAGRequest, RAGResponse               
)
from rag_api import RAG  # 核心RAG功能（语义嵌入、检索、生成回答等）

app = FastAPI()  # 创建FastAPI应用实例


# 语义嵌入接口
@app.post("/v1/embedding")
async def semantic_embedding(req: EmbeddingRequest) -> EmbeddingResponse:
    start_time = time.time()
    # 确保输入文本是列表（如果是单个字符串，转为长度1的列表）
    if not isinstance(req.text, list):
        text = [req.text]
    else:
        text = req.text

    # 调用RAG的方法生成文本的语义嵌入向量
    vector: np.ndarray = RAG().get_embedding(text)
    # 返回结果：把numpy数组转成列表（因为数组不能直接转JSON，列表可以）
    return EmbeddingResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),  # 向量转为列表返回（方便JSON序列化）
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time
    )


# 重排序接口
@app.post("/v1/rerank")
async def semantic_rerank(req: RerankRequest) -> RerankResponse:
    start_time = time.time()
    vector: np.ndarray = RAG().get_rank(req.text_pair)

    return RerankResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),  # 分数转为列表返回
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time
    )


rag = RAG()
# RAG聊天接口
@app.post("/chat")
def chat(req: RAGRequest) -> RAGResponse:
    start_time = time.time()
    message = rag.chat_with_rag(req.message)

    return RAGResponse(
        request_id=str(uuid.uuid4()),
        message=message,  # 生成的回答
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start_time
    )


if __name__ == "__main__":
    uvicorn.run(
        app,  
        host="0.0.0.0",  
        port=config["rag"]["port"], 
        workers=1 
    )

