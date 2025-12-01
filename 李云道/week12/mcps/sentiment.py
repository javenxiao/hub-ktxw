import os
from typing import Annotated
from fastmcp import FastMCP
from openai import AsyncOpenAI
import asyncio
import json
from enum import Enum

# 初始化FastMCP
mcp = FastMCP(
    name="Sentiment-MCP-Server",
    instructions="情感分析服务，提供文本情感分类和情感强度分析功能。",
)

# 初始化OpenAI客户端
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)


class SentimentLabel(str, Enum):
    """情感分类标签"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentIntensity(str, Enum):
    """情感强度"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@mcp.tool
async def sentiment_classification(
        text: Annotated[str, "需要分析情感的文本"],
        detailed: Annotated[bool, "是否返回详细分析", True] = True
) -> str:
    """对文本进行情感分类"""

    # 构建提示词
    prompt = f"""
请对以下文本进行情感分析：
"{text}"

请按照以下格式返回JSON结果：
{{
    "sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.0-1.0,
    "intensity": "weak|moderate|strong|very_strong",
    "explanation": "简要解释分析结果"
}}

{"如果detailed为true，还需要分析关键情感词和整体情感倾向" if detailed else ""}
"""

    try:
        # 调用OpenAI API
        response = await client.chat.completions.create(
            model="qwen-max",  # 可以根据需要改为gpt-4
            messages=[
                {"role": "system", "content": "你是一个专业的情感分析专家。请准确分析文本情感并返回规范的JSON格式结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度保证结果一致性
            response_format={"type": "json_object"}
        )

        # 解析响应
        result = response.choices[0].message.content
        sentiment_data = json.loads(result)

        # 构建返回结果
        if detailed:
            return f"""
情感分析结果：
- 情感分类: {sentiment_data.get('sentiment', 'unknown')}
- 置信度: {sentiment_data.get('confidence', 0) * 100:.1f}%
- 情感强度: {sentiment_data.get('intensity', 'unknown')}
- 分析说明: {sentiment_data.get('explanation', '无')}
"""
        else:
            return f"情感分类: {sentiment_data.get('sentiment', 'unknown')} (置信度: {sentiment_data.get('confidence', 0) * 100:.1f}%)"

    except Exception as e:
        return f"情感分析失败: {str(e)}"


@mcp.tool
async def sentiment_comparison(
        text1: Annotated[str, "第一段文本"],
        text2: Annotated[str, "第二段文本"]
) -> str:
    """比较两段文本的情感差异"""

    prompt = f"""
请比较以下两段文本的情感差异：

文本1: "{text1}"

文本2: "{text2}"

请返回JSON格式的比较结果：
{{
    "text1_sentiment": "positive|negative|neutral|mixed",
    "text2_sentiment": "positive|negative|neutral|mixed", 
    "sentiment_difference": "similar|slightly_different|very_different|opposite",
    "comparison_analysis": "详细比较分析",
    "overall_tendency": "哪段文本更积极/消极"
}}
"""

    try:
        response = await client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": "你是一个专业的情感分析专家，擅长比较不同文本的情感差异。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content
        comparison_data = json.loads(result)

        return f"""
情感比较结果：
📊 文本1情感: {comparison_data.get('text1_sentiment', 'unknown')}
📊 文本2情感: {comparison_data.get('text2_sentiment', 'unknown')}
🔍 情感差异: {comparison_data.get('sentiment_difference', 'unknown')}
📈 总体倾向: {comparison_data.get('overall_tendency', '未知')}

详细分析:
{comparison_data.get('comparison_analysis', '无')}
"""
    except Exception as e:
        return f"情感比较失败: {str(e)}"



@mcp.tool
async def emotional_trend_analysis(
        texts: Annotated[list, "按时间顺序的文本列表"],
        timeframe: Annotated[str, "时间范围描述", "近期"] = "近期"
) -> str:
    """分析情感趋势变化"""

    if len(texts) < 2:
        return "错误：至少需要2段文本来分析趋势"

    prompt = f"""
请分析以下{len(texts)}段文本的情感趋势变化（按时间顺序）：

{chr(10).join([f'时间段 {i + 1}: "{text}"' for i, text in enumerate(texts)])}

请分析情感趋势变化并返回JSON结果：
{{
    "trend_analysis": "情感变化趋势描述",
    "key_turning_points": ["关键转折点描述"],
    "overall_trend": "improving|deteriorating|stable|fluctuating",
    "recommendations": ["基于情感趋势的建议"]
}}
"""

    try:
        response = await client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": "你是一个专业的情感分析专家，擅长分析情感趋势变化。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content
        trend_data = json.loads(result)

        return f"""
📈 {timeframe}情感趋势分析：

趋势分析: {trend_data.get('trend_analysis', '无')}
总体趋势: {trend_data.get('overall_trend', '未知')}

🔍 关键转折点:
{chr(10).join(['• ' + point for point in trend_data.get('key_turning_points', [])]) or '无'}

💡 建议:
{chr(10).join(['• ' + rec for rec in trend_data.get('recommendations', [])]) or '无'}
"""
    except Exception as e:
        return f"情感趋势分析失败: {str(e)}"


# 运行服务器
if __name__ == "__main__":
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置OPENAI_API_KEY环境变量，使用默认密钥")

    # 启动MCP服务器
    print("启动情感分析MCP服务器...")
    print("服务端点: http://localhost:8903/sse")

    mcp.run(transport="sse", port=8903)