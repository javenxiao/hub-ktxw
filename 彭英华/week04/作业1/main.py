import traceback
from typing import Optional, Union
import time
import fastapi
from pydantic import BaseModel,Field
from bert import model_for_bert
class TextRequest(BaseModel):
    request_id: Optional[str] = Field(...,description="请求id,方便调试")
    request_text: Union[str,list[str]] = Field(...,description="需要分类的文本")
class TextResponse(BaseModel):
    request_id:Optional[str] = Field(...,description="请求id")
    request_text: Union[str,list[str]] = Field(...,description="需要分类的文本")
    classify_result: Union[str,list[str]] = Field(...,description="分类结果")
    classify_time: float = Field(...,description="分类耗时")
    err_msg: str = Field(...,description="异常信息")
app = fastapi.FastAPI()

@app.post("/classify_text/bert")
def text_for_bert(req:TextRequest) ->TextResponse:
    start_time = time.time()
    response = TextResponse(
        request_id = req.request_id,
        request_text = req.request_text,
        classify_result = "",
        classify_time=0,
        err_msg=""
    )
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.err_msg = "ok"
    except Exception as err:
        response.classify_result=""
        response.err_msg = traceback.format_exc()
    response.classify_time = round(time.time()-start_time,3)
    return response
