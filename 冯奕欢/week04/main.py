import fastapi
import time
import traceback
from data import TextClassifyRequest, TextClassifyResponse
from model.bert import bert_classify


app = fastapi.FastAPI()


@app.post("/v1/bert/classify")
def classify(request: TextClassifyRequest) -> TextClassifyResponse:
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        classify_result="",
        classify_text="",
        classify_time=0,
        error_msg=""
    )
    try:
        response.classify_result = bert_classify(request.request_text)
        response.classify_text = "好评" if response.classify_result == 1 else "差评"
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()
    response.classify_time = round(time.time() - start_time, 3)
    return response


