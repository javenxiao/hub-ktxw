import fastapi

from data import RecognitionRequest, RecognitionResponse
import recognition_by_prompt
import recognition_by_tools

app = fastapi.FastAPI()


@app.post("/v1/prompt")
def prompt(request: RecognitionRequest) -> RecognitionResponse:
    response = RecognitionResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        recognition_result=''
    )
    try:
        response.recognition_result = recognition_by_prompt.recognition(request.request_text)
    except Exception as err:
        print(err)
    return response


@app.post("/v1/tools")
def tools(request: RecognitionRequest) -> RecognitionResponse:
    response = RecognitionResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        recognition_result=''
    )
    try:
        response.recognition_result = recognition_by_tools.recognition(request.request_text)
    except Exception as err:
        print(err)
    return response


