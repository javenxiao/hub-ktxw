#-------部署模型---------
# python自带库
import time
import traceback
from typing import Union

# 第三方库
from fastapi import FastAPI

# 自己写的模块
from data_schema import TextClassifyResponse
from data_schema import TextClassifyRequest
from self_model.bert import model_for_bert
from logger import logger


app = FastAPI()

@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用TFIDF进行文本分类

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
    # info 日志
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response

#terminal定位main.py所在文件夹，执行fastapi run main.py，运行服务器

