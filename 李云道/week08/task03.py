import time
from typing import Literal, Union

import fastapi
from pydantic import BaseModel, Field

from config import HOST, PORT
from task02 import (
    do_prompt_process_with_openai,
    do_prompt_process_with_langchain,
    do_tools_process
)

'''
把作业2完成的提示词 + tools的代码，使用fastapi部署
'''


class NerRequest(BaseModel):
    """
    请求格式
    """
    request_id: str = Field(..., description="请求id")
    request_text: str = Field(..., description="请求文本")
    ner_type: Literal["prompt", "prompt-langchain", "tools"] = Field(..., description="命名实体识别类型")


class NerResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: str = Field(..., description="请求id")
    request_text: str = Field(..., description="请求文本")
    ner_type: Union[None | Literal["prompt", "prompt-langchain", "tools"]] = Field(..., description="命名实体识别类型")
    ner_result: str = Field(..., description="结果")
    ner_time: float = Field(..., description="命名实体识别耗时")
    msg: str = Field(..., description="信息")


app = fastapi.FastAPI()


@app.get("/")
@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/v1/ner/prompt")
def ner_prompt(request: NerRequest):
    start_time = time.time()
    response = NerResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        ner_time=0,
        msg="",
        ner_result="",
        ner_type=request.ner_type
    )
    try:
        if request.ner_type not in ["prompt", "prompt-langchain"]:
            response.msg = "error"
            response.ner_result = "ner_type error, only support 'prompt' or 'prompt-langchain'"
            return response
        if request.ner_type == "prompt":
            response.ner_result = do_prompt_process_with_openai(request.request_text)
        elif request.ner_type == "prompt-langchain":
            response.ner_result = do_prompt_process_with_langchain(request.request_text)
        response.ner_time = round(time.time() - start_time, 3)
        response.msg = "success"
    except Exception as e:
        response.msg = "error"
        response.ner_result = str(e)
        response.ner_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/ner/tools")
def ner_tools(request: NerRequest):
    start_time = time.time()
    response = NerResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        ner_time=0,
        msg="",
        ner_result="",
        ner_type=None
    )
    try:
        response.ner_result = do_tools_process(request.request_text)
        response.ner_time = round(time.time() - start_time, 3)
        response.msg = "success"
    except Exception as e:
        response.msg = "error"
        response.ner_result = str(e)
        response.ner_time = round(time.time() - start_time, 3)
    return response


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
