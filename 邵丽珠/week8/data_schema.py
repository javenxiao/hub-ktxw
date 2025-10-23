from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """
    请求格式
    """
    request_text: str = Field(..., description="请求文本、字符串或列表")


class TextResponse(BaseModel):
    """
    接口返回格式
    """
    intent: str = Field(..., description="意图识别")
    domain: str = Field(..., description="领域类别")
    slots: str = Field(..., description="实体标签")
