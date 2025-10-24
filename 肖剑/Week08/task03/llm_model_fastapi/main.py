import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    api_key="sk-4274c13a17904fff983a8c761c5bec9f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 系统提示词
SYSTEM_PROMPT = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / 
                cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / 
                LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / 
                dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / 
                location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / 
                tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / 
                endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / 
                theatre / datetime_time

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

{
    "domain": ,
    "intent": ,
    "slots": {
      "待选实体": "实体名词",
    }
}
"""


class ExtractionRequest(BaseModel):
    text: str = Field(..., description="需要抽取信息的文本")
    model_name: str = Field(default="qwen-plus", description="使用的模型名称")


class ExtractionResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    domain: Optional[str] = Field(None, description="领域")
    intent: Optional[str] = Field(None, description="意图")
    slots: Optional[Dict[str, Any]] = Field(None, description="实体槽位")
    error: Optional[str] = Field(None, description="错误信息")
    raw_response: Optional[str] = Field(None, description="原始响应内容")


# 创建 FastAPI 应用
app = FastAPI(
    title="信息抽取API",
    description="基于千问大模型的领域、意图和实体抽取服务",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_information(text: str, model_name: str = "qwen-plus") -> Dict[str, Any]:
    """
    调用千问模型进行信息抽取
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )

        response_content = completion.choices[0].message.content

        # 尝试解析 JSON 响应
        try:
            result_json = json.loads(response_content)
            return {
                "success": True,
                "domain": result_json.get("domain"),
                "intent": result_json.get("intent"),
                "slots": result_json.get("slots", {}),
                "raw_response": response_content
            }
        except json.JSONDecodeError:
            # 如果返回的不是合法 JSON，尝试从文本中提取
            return {
                "success": False,
                "error": "API返回的内容不是有效的JSON",
                "raw_response": response_content
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"API调用失败: {str(e)}",
            "raw_response": None
        }


@app.get("/")
async def root():
    """根端点，返回服务信息"""
    return {
        "service": "信息抽取API",
        "version": "1.0.0",
        "description": "基于千问大模型的领域、意图和实体抽取服务",
        "endpoints": {
            "extract": "POST /extract - 单个文本抽取",
            "batch_extract": "POST /batch_extract - 批量文本抽取",
            "test": "GET /test - 测试服务"
        }
    }


@app.get("/test")
async def test_service():
    """测试服务端点"""
    test_cases = [
        "预订从许昌到中山的班车。",
        "播放周杰伦的青花瓷",
        "查询上海今天的天气",
        "搜索最近的餐厅",
        "帮我翻译'你好'成英文"
    ]

    results = []
    for text in test_cases:
        result = extract_information(text)
        results.append({
            "text": text,
            "success": result["success"],
            "domain": result.get("domain"),
            "intent": result.get("intent"),
            "slots": result.get("slots"),
            "error": result.get("error")
        })

    return {
        "service_status": "running",
        "test_results": results
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=True
    )