import asyncio

import mcp
from fastmcp import FastMCP, Client

from news import mcp as news_mcp
from tool import mcp as tool_mcp
from saying import mcp as saying_mcp
from classification import mcp as classification_mcp


# MCP实例
mcp = FastMCP(name="All-MCP-Server")


async def setup():
    """集成多个MCP服务"""
    await mcp.import_server(news_mcp, prefix="news")
    await mcp.import_server(tool_mcp, prefix="tool")
    await mcp.import_server(saying_mcp, prefix="saying")
    await mcp.import_server(classification_mcp, prefix="classification")


async def show_all_mcp():
    """显示所有支持的MCP工具方法"""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("all tool: ", [tool.name for tool in tools])


if __name__ == '__main__':
    asyncio.run(setup())
    asyncio.run(show_all_mcp())
    mcp.run(transport="sse", port=8900)
