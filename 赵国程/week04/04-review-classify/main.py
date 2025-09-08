import time
import traceback
from fastapi import FastAPI

from data_schema import TextClassifyResponse, TextClassifyRequest
from logger import logger
from model.bery_pred import model_for_bert

app = FastAPI()

@app.post("/review-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    使用bert进行分类
    :param req:
    :return:
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id = req.request_id,
        request_text = req.request_text,
        classify_result = "",
        classify_time=0,
        error_msg=""
    )

    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as e:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response
