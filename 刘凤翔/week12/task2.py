from typing import Annotated
import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

# 添加情感分析工具
@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text into positive, negative or neutral."""
    try:
        # 这里可以使用现有的API或实现自己的情感分析逻辑
        # 示例：使用现有的API服务
        response = requests.get(f"https://whyta.cn/api/tx/nlp/sentiment?key={TOKEN}&text={text}")
        if response.status_code == 200:
            result = response.json()
            return result.get("result", "分析失败")
        else:
            # 如果API不可用，使用简单的基于关键词的情感分析
            return simple_sentiment_analysis(text)
    except:
        # 备用方案：基于关键词的简单情感分析
        return simple_sentiment_analysis(text)

def simple_sentiment_analysis(text):
    """简单的基于关键词的情感分析"""
    positive_words = ['好', '喜欢', '开心', '高兴', '满意', '优秀', '棒', '赞', '美丽', '漂亮']
    negative_words = ['差', '讨厌', '生气', '愤怒', '失望', '糟糕', '烂', '差劲', '丑陋', '恶心']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return {"sentiment": "正面", "confidence": min(positive_count/10, 1.0), "reason": f"检测到{positive_count}个正面词汇"}
    elif negative_count > positive_count:
        return {"sentiment": "负面", "confidence": min(negative_count/10, 1.0), "reason": f"检测到{negative_count}个负面词汇"}
    else:
        return {"sentiment": "中性", "confidence": 0.5, "reason": "正负面词汇数量相当"}