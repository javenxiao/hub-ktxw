from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional


class TextClassifyRequest(BaseModel):
    """
    接口请求格式
    """
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Optional[str] = Field(..., description="请求文本")


class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Optional[str] = Field(..., description="请求文本")
    classify_result: Optional[str] = Field(..., description="分类结果数值")
    classify_text: Optional[str] = Field(..., description="分类结果文字")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")