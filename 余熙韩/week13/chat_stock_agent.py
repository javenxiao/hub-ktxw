from typing import Any


import os
from agents import Agent, Handoff, ModelSettings
from api.autostock import app
from fastmcp import FastMCP, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


os.environ["OPENAI_API_KEY"] = "sk-0fce7eb19bc042469a27ea5005b65635"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

autostock_mcp = FastMCP.from_fastapi(app=app)

external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
# 闲聊 Agent：处理日常寒暄、问候、天气等无关股票的内容
chat_agent = Agent(name="chat_agent",
                   instructions="你是一个友好的闲聊助手，可以回答天气、问候、日常话题等，不涉及股票信息。",
                    model=OpenAIChatCompletionsModel(
                        model="qwen-max",
                        openai_client=external_client,
                    ),
                    model_settings=ModelSettings(parallel_tool_calls=False) # 禁用并行工具调用，因为股票qwen不支持
                   )

# 股票 Agent：专门处理股票相关查询
stock_agent = Agent(name="stock_agent",
                    instructions="你是一个专业的股票助手，能够回答股票行情、分析、指标等相关问题。",
                    mcp_servers=[autostock_mcp],
                    model=OpenAIChatCompletionsModel(
                        model="qwen-max",
                        openai_client=external_client,
                    ),
                    model_settings=ModelSettings(parallel_tool_calls=False) # 禁用并行工具调用
            )



triage_agent = Agent(name="triage_agent",
                     instructions="你是一个路由助手，根据用户输入判断是否需要股票相关查询。",
                     model=OpenAIChatCompletionsModel(
                        model="qwen-max",
                        openai_client=external_client,
                    ),
                    model_settings=ModelSettings(parallel_tool_calls=False) # 禁用并行工具调用
            )
    
