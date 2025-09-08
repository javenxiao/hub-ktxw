from typing import Union

from pydantic import BaseModel, Field


class TextClassifyRequest(BaseModel):
    """
    请求格式
    """
    request_id: Union[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, list[str]] = Field(..., description="请求文本、字符串或列表")


class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Union[str] = Field(..., description="请求id")
    request_text: Union[str, list[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, list[str]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    msg: str = Field(..., description="信息")
