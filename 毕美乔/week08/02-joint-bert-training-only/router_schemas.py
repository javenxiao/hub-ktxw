from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class QueryRequest(BaseModel):
    text: str = Field(..., description="用户输入的文本")


class QueryResponse(BaseModel):
    text: str
    domain: str
    intent: str
    slots: Dict[str, str]