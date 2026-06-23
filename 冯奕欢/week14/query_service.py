import asyncio
import os

from rag_service import search

from agents.mcp.util import ToolFilterStatic
from agents.mcp.server import MCPServerSse
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel


os.environ["OPENAI_API_KEY"] = "sk-b15ba6a031b447f7b18d195b834629f1"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


async def ask_question(question):
    """
    回答问题
    :param question: 问题
    :return: 回答
    """
    # RAG获取工具
    results = search(question)
    tools = [result['tool'] for result in results]
    print(tools)
    # Agent Tool
    tool_filter = ToolFilterStatic(allowed_tool_names=tools)
    # 连接 MCP Service
    async with MCPServerSse(
        name="MCP Sse Server",
        params={
            "url": "http://localhost:8900/sse"
        },
        tool_filter=tool_filter,
        client_session_timeout_seconds=20
    ) as mcp_server:
        # Agent
        external_client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )
        agent = Agent(
            name="Assistant",
            instructions="使用工具计算",
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model="qwen-max",
                openai_client=external_client,
            )
        )
        # 回答
        result = await Runner.run(starting_agent=agent, input=question)
        print(result.final_output)
        return result.final_output


if __name__ == '__main__':
    question = "在水温25℃、饲料投喂率3.0%、溶解氧6.0mg/L的养殖条件下，鱼类的日增重是多少？"
    result = asyncio.run(ask_question(question))
    print(result)