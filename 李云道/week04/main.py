import time
import traceback

from fastapi import FastAPI

from transdata import TextClassifyRequest, TextClassifyResponse
from logger import logger
from model.bert import model_for_bert

app = FastAPI()


@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用BERT进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        msg=""
    )
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.msg = "ok"
    except Exception as e:
        response.classify_result = ""
        response.msg = traceback.format_exc()
    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.get("/ping")
def ping():
    return {"msg": "pong"}


if __name__ == '__main__':
    import uvicorn
    from config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT)
