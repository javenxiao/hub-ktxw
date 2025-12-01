from typing import Annotated

TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Sentiment-MCP-Server",
    instructions="""This server can determine the sentiment category of the text.""",
)
@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    pass
