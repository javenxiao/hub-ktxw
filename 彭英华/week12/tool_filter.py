from agents import Agent,Runner,set_default_openai_api,set_tracing_disabled
from agents.mcp import MCPServer
from agents.mcp.server import MCPServerSse
from agents import AsyncOpenAI,OpenAIChatCompletionsModel
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
from agents.mcp import ToolFilterStatic,ToolFilterCallable,ToolFilterContext
from mcp.types import Tool as MCPTool
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
intent = input("输入你要做什么")
news_keywords = ['新闻','今日头条','日常新闻','抖音热点','github','时事热点','体育新闻']
#动态配置查询新闻的时候，只调用news的工具
async def news_tool_filter(ctx:ToolFilterContext,tool:MCPTool):
    allowed_name =['get_today_daily_news','get_douyin_hot_news','get_github_hot_news','get_toutiao_hot_news','get_sports_news']
    for keyword in newskeywords:
        if (ctx.run_context and ctx.run_context.context and keyword in ctx.run_context.context):
            if tool.name in allowed_name:
                return True
            else:
                return False
        else:
            return True
#静态配置只可调用tools的工具
static_tool_filter = ToolFilterStatic(allowed_tool_names=['get_city_weather','get_address_detail','get_tel_info','get_scenic_info','get_flower_info','get_rate_transform'])
async def run(mcp_server:MCPServer):
    externel_client = AsyncOpenAI(api_key="sk-90aa2b7df82745f3a46373cc0ddd0497",
                                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    agent = Agent(name="my agent",instructions="调用可用的工具完成任务",
                  mcp_servers=[mcp_server],
                  model = OpenAIChatCompletionsModel(model="qwen-flash",openai_client=externel_client))
    message="结合工具对‘今天的新闻联播并没有按时播放’这句话进行情感分析,并给出工具的回答"
    result=  Runner.run_streamed(agent,input=message,context=intent)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
async def main():
    async with MCPServerSse(name="SSE Server Python",
                            params={
                                "url":"http://127.0.0.1:11314/sse",
                            },
                            # tool_filter=static_tool_filter,              #配置工具过滤逻辑
                            client_session_timeout_seconds=20)as server:
        await run(server)
if __name__ == "__main__":
    asyncio.run(main())
