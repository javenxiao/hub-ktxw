from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json
import re

# 初始化 FastAPI 应用
app = FastAPI(
    title="信息抽取API",
    description="专业信息抽取服务，抽取文本的领域类别、意图类型和实体标签",
    version="1.0.0"
)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-ba6c90b8843d4ccd993c12fcfd2893b3",
    base_url="https://api.deepseek.com"
)

# 定义请求模型


class ExtractRequest(BaseModel):
    text: str
    model: str = "deepseek-chat"  # 默认使用非思考模式

# 定义响应模型


class ExtractResponse(BaseModel):
    domain: str
    intent: str
    slots: dict


@app.post("/extract", response_model=ExtractResponse, summary="信息抽取")
async def extract_info(request: ExtractRequest):
    """
    对输入文本进行信息抽取，返回领域类别、意图类型和实体标签

    - **text**: 需要抽取信息的文本
    - **model**: 模型类型，deepseek-chat(非思考模式) 或 deepseek-reasoner(思考模式)
    """
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

```json
{
    "domain": ,
    "intent": ,
    "slots": {
        "待选实体": "实体名词",
    }
}
```"""},
                {"role": "user", "content": request.text},
            ],
            stream=False
        )

        # 解析响应内容
        content = response.choices[0].message.content

        # 提取 JSON 部分（处理可能的额外文本）
        try:
            # 尝试直接解析整个响应
            result = json.loads(content)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取 JSON 部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("无法从响应中提取有效的 JSON 数据")

        return ExtractResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信息抽取失败: {str(e)}")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": "信息抽取API服务",
        "version": "1.0.0",
        "endpoints": {
            "POST /extract": "信息抽取接口"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}
