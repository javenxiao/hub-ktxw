import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    api_key="sk-4274c13a17904fff983a8c761c5bec9f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # 获取模型的 JSON schema
        schema = response_model.model_json_schema()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get('title', 'extraction_function'),
                    "description": schema.get('description', 'Extract information from text'),
                    "parameters": {
                        "type": "object",
                        "properties": schema.get('properties', {}),
                    },
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            if (response.choices[0].message.tool_calls and
                    len(response.choices[0].message.tool_calls) > 0):
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                return response_model.model_validate_json(arguments)
            else:
                print("No tool calls in response")
                return None

        except Exception as e:
            print(f"API调用错误: {e}")
            return None


# 定义领域和意图的常量，单独定义避免过长的行
DOMAINS = Literal[
    'music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather', 'match',
    'map', 'website', 'news', 'message', 'contacts', 'translation', 'tvchannel',
    'cinemas', 'cookbook', 'joke', 'riddle', 'telephone', 'video', 'train',
    'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'
]

INTENTS = Literal[
    'OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY',
    'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY',
    'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT',
    'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'
]


class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: DOMAINS = Field(description="领域")
    intent: INTENTS = Field(description="意图")
    slots: Dict[str, Any] = Field(description="实体槽位字典", default_factory=dict)


# 定义请求和响应模型
class ExtractionRequest(BaseModel):
    text: str = Field(..., description="需要抽取信息的文本")
    model_name: str = Field(default="qwen-plus", description="使用的模型名称")


class ExtractionResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    domain: Optional[str] = Field(None, description="领域")
    intent: Optional[str] = Field(None, description="意图")
    slots: Optional[Dict[str, Any]] = Field(None, description="实体槽位")
    error: Optional[str] = Field(None, description="错误信息")


# 创建 FastAPI 应用
app = FastAPI(
    title="信息抽取API",
    description="基于大模型的领域、意图和实体抽取服务",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 agent 实例
agent = ExtractionAgent(model_name="qwen-plus")


@app.get("/")
async def root():
    """根端点，返回服务信息"""
    return {
        "service": "信息抽取API",
        "version": "1.0.0",
        "description": "基于大模型的领域、意图和实体抽取服务"
    }


# 测试端点
@app.get("/test")
async def test_endpoint():
    """测试端点"""
    test_cases = [
        "帮我查询下从北京到天津到武汉的汽车票",
        "播放周杰伦的青花瓷",
        "查询上海今天的天气"
    ]

    results = []
    for text in test_cases:
        result = agent.call(text, IntentDomainNerTask)
        if result:
            results.append({
                "text": text,
                "domain": result.domain,
                "intent": result.intent,
                "slots": result.slots
            })
        else:
            results.append({
                "text": text,
                "error": "抽取失败"
            })

    return {"test_results": results}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # 这里必须是 "main:app"
        host="127.0.0.1",
        port=8000,
        reload=True
    )