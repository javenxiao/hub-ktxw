from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional


class TextClassifyRequest(BaseModel):
    """
    请求格式
    """
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_text: Union[str, List[str]] = Field(..., description="输入文本")
    classify_result: List[int] = Field(..., description="分类结果")
    classify_meaning: List[str] = Field(..., description="含义")
    error_msg: str = Field(..., description="异常信息")
