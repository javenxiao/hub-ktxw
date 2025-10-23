from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import json
import re
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

# 配置
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-ba6c90b8843d4ccd993c12fcfd2893b3")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coze 信息抽取API")

# CORS 配置 - 允许 Coze 调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# 请求响应模型
class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    model: str = Field(default="deepseek-chat")


class ExtractResponse(BaseModel):
    domain: str
    intent: str
    slots: Dict[str, Any]
    confidence: Optional[float] = None  # 为 Coze 添加置信度


class CozeWebhookRequest(BaseModel):
    """Coze Webhook 专用请求格式"""
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class CozeWebhookResponse(BaseModel):
    """Coze Webhook 专用响应格式"""
    reply: str
    extracted_data: Optional[Dict[str, Any]] = None
    success: bool = True


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": "信息抽取API服务",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract": "信息抽取接口",
            "GET /docs": "API文档",
            "GET /health": "健康检查"
        }
    }

# Coze 专用端点
@app.post("/coze/webhook", response_model=CozeWebhookResponse)
async def coze_webhook(request: CozeWebhookRequest):
    """
    Coze Webhook 专用接口
    接收用户消息，返回格式化回复
    """
    try:
        logger.info(f"Coze Webhook 请求: {request.message}")

        # 调用信息抽取
        extract_request = ExtractRequest(text=request.message)
        extraction_result = await extract_info_internal(extract_request)

        # 构建 Coze 友好的回复
        reply = format_coze_reply(extraction_result)

        return CozeWebhookResponse(
            reply=reply,
            extracted_data=extraction_result.dict(),
            success=True
        )

    except Exception as e:
        logger.error(f"Coze Webhook 处理失败: {str(e)}")
        return CozeWebhookResponse(
            reply="抱歉，信息处理出现了一些问题，请稍后重试。",
            success=False
        )


@app.post("/extract", response_model=ExtractResponse)
async def extract_info(request: ExtractRequest):
    """原始信息抽取接口"""
    return await extract_info_internal(request)


async def extract_info_internal(request: ExtractRequest) -> ExtractResponse:
    """内部处理逻辑"""
    try:
        system_prompt = """你是一个专业信息抽取专家..."""  # 你的原有提示词

        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text},
            ],
            stream=False
        )

        content = response.choices[0].message.content
        result = extract_json_from_text(content)

        return ExtractResponse(**result)

    except Exception as e:
        logger.error(f"信息抽取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


def format_coze_reply(extraction_result: ExtractResponse) -> str:
    """为 Coze 构建用户友好的回复"""
    result = extraction_result.dict()

    reply = "**信息分析结果**\n\n"
    reply += f"**领域分类**：{result['domain']}\n"
    reply += f"**意图识别**：{result['intent']}\n"

    if result['slots']:
        reply += "\n**提取的关键信息**：\n"
        for key, value in result['slots'].items():
            if value:
                reply += f"• {key}: {value}\n"
    else:
        reply += "\n未识别到具体实体信息"


    return reply


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """提取 JSON 内容"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"domain": "unknown", "intent": "DEFAULT", "slots": {}}


# 健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Coze Information Extraction API",
        "timestamp": datetime.now().isoformat()
    }