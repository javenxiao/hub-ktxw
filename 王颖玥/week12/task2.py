from typing import Annotated, Union
import requests
from openai import AsyncOpenAI
import os
import json

TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)


@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    # 功能说明：通过城市拼音获取当前天气数据
    """Retrieves the current weather data using the city's Pinyin name.[type: tool]"""
    try:
        # 调用天气API，传入城市拼音参数；解析返回的JSON数据中的"data"字段（天气详情）
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []


@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    # 功能说明：解析地址字符串，提取省/市/区等详细信息
    """Parses a raw address string to extract detailed components (province, city, district, etc.).[type: tool]"""
    try:
        # 调用地址解析API，传入地址文本；解析返回的JSON数据中的"result"字段
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []


@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    # 功能说明：获取电话号码的基本信息（归属地、运营商等）
    """Retrieves basic information (location, carrier) for a given telephone number.[type: tool]"""
    try:
        # 调用手机号信息API，传入电话号码；解析返回的JSON数据中的"result"字段
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []


@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    # 功能说明：查询特定景点或旅游景点的信息
    """Searches for and retrieves information about a specific scenic spot or tourist attraction.[type: tool]"""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []


@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name.[type: tool]"""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []


@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    # 功能说明：计算两种货币之间的兑换金额
    """Calculates the currency exchange conversion amount between two specified coins.[type: tool]"""
    # 参数说明：
    # source_coin：字符串，源货币的3字母代码（如USD、CNY）
    # aim_coin：字符串，目标货币的3字母代码（如EUR、JPY）
    # money：int或float类型，要兑换的金额
    try:
        # 调用汇率转换API，传入源货币、目标货币和金额；解析返回的JSON数据中的"result"->"money"字段（兑换结果）
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []


os.environ["OPENAI_API_KEY"] = "sk-b8f16de2371547c397349552ecffce68"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"]
)


@mcp.tool
async def sentiment_classification(text: Annotated[str, "Sentence requiring emotional analysis"]):
    """Analyze the emotional tendency of the text.[type: tool]"""
    try:
        completion = await client.chat.completions.create(
            model="qwen-flash",
            messages=[
                {"role": "system",
                 "content": """
                            你是一个文本情感分析专家，请分析以下文本的情感倾向，严格按照以下要求输出：
                            1. 情感类型只能是 positive，negative，neutral 这三种中的一种；
                            2. 只输出分类结果，不要有其他任何解释；
                            3. 要求最终输出的格式是如下的json格式，每一个文本都按照如下格式输出：
                            {
                                "text": "我今天要和妈妈一起去吃冒菜！",
                                "sentiment_type": "positive" 
                            }
                            """
                 },
                {"role": "user", "content": text}
            ]
        )
        result = completion.choices[0].message.content.strip()
        result_dict = json.loads(result)
        return {
            "text": text,
            "sentiment_type": result_dict["sentiment_type"]
        }

    except Exception as e:
        raise {"error": str(e), "sentiment_type": "unknown"}


# 干扰测试
@mcp.tool
def fake_news_tool(text: Annotated[str, "False news tools"]):
    """Retrieves a list of today's daily news bulletin items from the external API.[type: tool]"""
    return {"error": "调用错误！"}
