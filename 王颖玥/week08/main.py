from fastapi import FastAPI
from prompt_tools import tools_extraction, prompt_Extraction
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional

app = FastAPI(title="抽取领域类别、意图类别和实体标签")


class TextClassifyRequest(BaseModel):
    request_text: Union[str, List[str]] = Field(..., description="需要分析的文本或文本列表")


@app.post("/prompt")
def prompt(req: TextClassifyRequest):
    result = prompt_Extraction(req.request_text)
    return result


@app.post("/tools")
def tools(req: TextClassifyRequest):
    result = tools_extraction(req.request_text)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="127.0.0.1", port=8000, reload=True)

