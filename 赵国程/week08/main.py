import time
import traceback
from fastapi import FastAPI

from data_schema import TextExtractResponse, TextExtractRequest
from extract_tools import extract_tools
from extract_prompt import extract_prompt

app = FastAPI()

@app.post("/extract/tools")
def text_extract(req: TextExtractRequest) -> TextExtractResponse:

    start_time = time.time()

    response = TextExtractResponse(
        request_id = req.request_id,
        request_text = req.request_text,
        error_msg="",
        extract_result={},
        extract_time=0.0
    )

    try:
        response.extract_result = extract_tools(req.request_text)
        response.error_msg = "ok"
    except Exception as e:
        response.error_msg = traceback.format_exc()

    response.extract_time = round(time.time() - start_time, 3)
    return response

@app.post("/extract/prompt")
def text_extract(req: TextExtractRequest) -> TextExtractResponse:

    start_time = time.time()

    response = TextExtractResponse(
        request_id = req.request_id,
        request_text = req.request_text,
        error_msg="",
        extract_result={},
        extract_time=0.0
    )

    try:
        response.extract_result = extract_prompt(req.request_text)
        response.error_msg = "ok"
    except Exception as e:
        response.error_msg = traceback.format_exc()

    response.extract_time = round(time.time() - start_time, 3)
    return response

