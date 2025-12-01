import requests
from typing import Annotated

TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Sentiment-MCP-Server",
    instructions="""This server contains some api of sentiment.""",
)

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    try:
        return requests.get(f"https://hanlp.hankcs.com/demos/sentiment.html?text={text}").json()["result"]["list"]
    except:
        return []