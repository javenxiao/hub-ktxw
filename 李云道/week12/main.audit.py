import asyncio
from fastmcp import FastMCP, Client

from mcps.news import mcp as mcps_news
from mcps.saying import mcp as mcps_saying
from mcps.tool import mcp as mcps_tool
from mcps.sentiment import mcp as mcps_sentiment

mcp = FastMCP(
    name="MCP-Server"
)

async def setup():
    await mcp.import_server(mcps_news, prefix="")
    await mcp.import_server(mcps_saying, prefix="")
    await mcp.import_server(mcps_tool, prefix="")
    await mcp.import_server(mcps_sentiment, prefix="")

async def t_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(t_filtering())
    mcp.run(transport="sse", port=8900)
