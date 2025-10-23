from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional


class RecognitionRequest(BaseModel):
    """
    接口请求格式
    """
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Optional[str] = Field(..., description="请求文本")


class RecognitionResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Optional[str] = Field(..., description="请求文本")
    recognition_result: Optional[str] = Field(..., description="识别结果")