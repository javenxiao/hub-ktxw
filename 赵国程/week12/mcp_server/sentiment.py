from fastmcp import FastMCP
from fontTools.ttLib.tables.ttProgram import instructions
from openai import OpenAI
import base64

mcp = FastMCP(
    name="Sentiment-MCP-Server",
    instructions="你是一个情感分析模型，对输入的文本进行情感分类"
)

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-88d3ca9887854f7e92a8485baa49993f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@mcp.tool
async def sentiment_analysis(text: str) -> str:
    """对输入的文本进行情感分类"""
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"请对以下文本进行情感分类：{text}"}
                ],
            },
        ],
        stream=False,
        extra_body={
            'enable_thinking': True,
            "thinking_budget": 81920},
    )
    return completion.choices[0].message.content.strip()

