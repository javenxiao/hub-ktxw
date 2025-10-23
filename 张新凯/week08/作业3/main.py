# python自带库
import time
import traceback

# 第三方库
from fastapi import FastAPI

# 自己写的模块
from data_schema import TextClassifyRequest
from data_schema import TextClassifyResponse
from logger import logger
from model.prompt import classify_by_prompt
from model.tools import classify_by_tools

app = FastAPI()


@app.post("/v1/text-cls/tools")
def tools_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用tools进行意图识别

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result=dict(),
        classify_time=0,
        error_msg=""
    )
    # info 日志

    logger.info(f"{req.request_id} {req.request_text}")  # 打印请求
    try:
        response.classify_result = classify_by_tools(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/text-cls/prompt")
def prompt_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用prompt进行意图识别

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result=dict(),
        classify_time=0,
        error_msg=""
    )
    # info 日志

    logger.info(f"{req.request_id} {req.request_text}")  # 打印请求
    try:
        response.classify_result = classify_by_prompt(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
