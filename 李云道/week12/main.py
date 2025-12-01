import time
import threading

from fastmcp import Client,FastMCP
from mcps.news import mcp as mcps_news
from mcps.saying import mcp as mcps_saying
from mcps.tool import mcp as mcps_tool
from mcps.sentiment import mcp as mcps_sentiment


def run_server(mcp, port, name):
    """在单独的线程中运行服务器"""
    print(f"启动 {name} 在端口 {port}")
    try:
        mcp.run(transport="sse", port=port)
    except Exception as e:
        print(f"{name} 启动失败: {e}")


def main():
    # 在单独的线程中启动每个服务器
    threads = []
    servers = [
        (mcps_news, 8900, "新闻服务"),
        (mcps_saying, 8901, "名言服务"),
        (mcps_tool, 8902, "工具服务"),
        (mcps_sentiment, 8903, "情感分析服务")
    ]

    for mcp, port, name in servers:
        thread = threading.Thread(target=run_server, args=(mcp, port, name), daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(1)  # 给每个服务器启动时间

    print("所有服务已启动，按 Ctrl+C 停止")

    try:
        # 主线程保持运行
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n正在停止服务...")


if __name__ == '__main__':
    main()
