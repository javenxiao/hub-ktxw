# python自带库
import time
import traceback
from typing import Union

# 第三方库
import openai
from fastapi import FastAPI

# 自己写的模块
from data_schema import TextClassifyResponse
from data_schema import TextClassifyRequest
from llm.prompt import classify_by_prompt
from llm.tools import classify_by_tools

from logger import logger

#print(classify_by_tools("帮我查询下从北京到天津到武汉的汽车票"))
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/v1/text-cls/prompt")
def classify_prompt(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用大语言模型进行文本分类, 使用提示词

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    logger.info(f"{req.request_id} {req.request_text}") # 打印请求
    try:
        response.classify_result = classify_by_prompt(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/tools")
def classify_tools(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用大语言模型进行文本分类, 使用tools

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    logger.info(f"Get requst: {req.json()}")

    try:
        response.classify_result = classify_by_tools(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response
