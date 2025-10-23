from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional


class TextExtractRequest(BaseModel):
    """
    请求格式
    """
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


class TextExtractResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    extract_result: Union[Dict, List[Dict]] = Field(..., description="分类结果")
    extract_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")
