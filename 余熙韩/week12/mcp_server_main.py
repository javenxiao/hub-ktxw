import asyncio
from fastmcp import FastMCP, Client
from fastmcp.client.transports import FastMCPTransport

from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp
from sentiment_classification import mcp as sentiment_classification_mcp


# 创建主MCP服务器（包含所有工具）
mcp = FastMCP(
    name="Main-MCP-Server"
)

# 创建专门用于新闻的MCP服务器
news_mcp_server = FastMCP(
    name="News-MCP-Server"
)

# 创建专门用于工具的MCP服务器
tools_mcp_server = FastMCP(
    name="Tools-MCP-Server"
)


async def setup():
    # 主服务器导入所有工具
    await mcp.import_server(news_mcp, prefix="news")
    await mcp.import_server(saying_mcp, prefix="saying")
    await mcp.import_server(tool_mcp, prefix="tools")
    await mcp.import_server(sentiment_classification_mcp, prefix="sentiment")
    
    # 新闻服务器只导入新闻工具
    await news_mcp_server.import_server(news_mcp, prefix="news")
    
    # 工具服务器只导入工具类工具
    await tools_mcp_server.import_server(tool_mcp, prefix="tools")


async def test_filtering():
    # 测试主服务器的工具列表
    async with Client[FastMCPTransport](mcp) as client:
        tools = await client.list_tools()
        print("Main server tools:", [t.name for t in tools])
    
    # 测试新闻服务器的工具列表
    async with Client[FastMCPTransport](news_mcp_server) as client:
        tools = await client.list_tools()
        print("News-only server tools:", [t.name for t in tools])
    
    # 测试工具服务器的工具列表
    async with Client[FastMCPTransport](tools_mcp_server) as client:
        tools = await client.list_tools()
        print("Tools-only server tools:", [t.name for t in tools])


if __name__ == "__main__":
    asyncio.run(setup())  # 运行 setup 函数，导入所有的mcp服务
    asyncio.run(test_filtering())  # 运行 test_filtering 函数，测试所有的mcp服务
    
    print("\nStarting MCP servers...")
    print("Main server running on port 8900")
    print("News-only server running on port 8901")
    print("Tools-only server running on port 8902")
    
    # 定义启动服务器的异步函数 # 启动3个mcpserver  调用的时候根据提示词，选择不同的mcpserver
    async def start_servers():
        # 创建并启动所有服务器
        tasks = [
            asyncio.create_task(mcp.run_sse_async(port=8900)),
            asyncio.create_task(news_mcp_server.run_sse_async(port=8901)),
            asyncio.create_task(tools_mcp_server.run_sse_async(port=8902))
        ]
        
        # 等待所有服务器完成
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\nServers stopped by user")
    
    # 启动服务器
    try:
        asyncio.run(start_servers())
    except KeyboardInterrupt:
        print("\nServers stopped by user")
