import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-0fce7eb19bc042469a27ea5005b65635"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
from fastmcp import FastMCP
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)



mcp = FastMCP(
    name="Sentiment-Classification-MCP-Server",
    instructions="""This server contains some api of sentiment classification.""",
)


# 复用客户端与模型，避免每次调用都新建
_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
_model = OpenAIChatCompletionsModel(model="qwen-plus", openai_client=_client)

# 创建Agent
agent = Agent(
    name="sentiment_classifier",
    instructions="You are a sentiment classification model. "
    "Please only return one word: positive, negative, or neutral.",
    model=_model,
)


# 把 agent 当作一个工具，注册到 MCP 服务器，
# 这样在调用 MCP 服务器时，就可以直接调用这个工具
@mcp.tool
async def classify_sentiment(text: str) -> str:
    """
    对文本进行情感分类
    
    Args:
        text: 待分类的文本
        
    Returns:
        str: 情感分类结果（如 "positive", "negative", "neutral"）
    """

    # 运行Agent
    result = await Runner.run(
        agent, f"Classify the sentiment of the following text: {text}")
    # print(result.final_output)
    return result.final_output.strip()
