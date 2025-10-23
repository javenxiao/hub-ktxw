import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Optional,Union
from prompt import prompt_llm
from llm_tools import tools_llm
import time
import traceback
class TextRequest(BaseModel):
    request_id:Optional[str] = Field(...,description="请求id")
    request_text:Union[str,list[str]] = Field(...,description="待进行领域+意图+实体抽取的文本")
    model_name:str = Field(...,description="使用模型名称")
class TextResponse(BaseModel):
    request_id: Optional[str] = Field(...,description="请求id")
    request_text: Union[str, list[str]] = Field(...,description="待进行领域+意图+实体抽取的文本")
    extract_result: Union[str,list[dict]] = Field(..., description="识别结果")
    extract_time: float = Field(..., description="识别耗时")
    err_msg: str = Field(..., description="异常信息")
app = FastAPI()
@app.post("/extract/llm_prompt")
def extract_for_prompt(req:TextRequest) ->TextResponse:
    start_time = time.time()
    response = TextResponse(request_id =req.request_id,
                            request_text = req.request_text,
                            extract_result = "",
                            extract_time = 0,
                            err_msg = " ")
    try:
        response.extract_result = prompt_llm(req.request_text,req.model_name)
        response.err_msg = "ok"
    except Exception as err:
        response.extract_result = ""
        response.err_msg = traceback.format_exc()
    response.extract_time = round(time.time()-start_time,3)
    return response
@app.post("/extract/llm_tools")
def extract_for_tools(req:TextRequest) ->TextResponse:
    start_time = time.time()
    response = TextResponse(request_id=req.request_id,
                            request_text=req.request_text,
                            extract_result="",
                            extract_time=0,
                            err_msg=" ")
    try:
        response.extract_result = tools_llm(req.request_text, req.model_name)
        response.err_msg = "ok"
    except Exception as err:
        response.extract_result = ""
        response.err_msg = traceback.format_exc()
    response.extract_time = round(time.time() - start_time, 3)
    return response
if __name__ == "__main__":
    uvicorn.run(app,port=11434)
