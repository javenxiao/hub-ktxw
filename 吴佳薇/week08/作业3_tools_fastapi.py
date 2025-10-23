from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal
import openai
import uvicorn

# 初始化 FastAPI 应用
app = FastAPI(
    title="信息抽取 API",
    description="基于大语言模型的领域、意图和实体抽取服务",
    version="1.0.0"
)

# OpenAI 客户端配置
client = openai.OpenAI(
    api_key="sk-4cb91fbab0154645a837fe4afb5d35ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 领域、意图、实体选项定义
text_domain = 'music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story'
text_intent = 'OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION'
text_slot = 'code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time'

# 清理数据并创建Literal选项
domain_options = [item.strip() for item in text_domain.split('/') if item.strip()]
intent_options = [item.strip() for item in text_intent.split('/') if item.strip()]
slot_options = [item.strip() for item in text_slot.split('/') if item.strip()]


# 信息抽取智能体
class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [{"role": "user", "content": user_prompt}]

        tools = [{
            "type": "function",
            "function": {
                "name": response_model.model_json_schema()['title'],
                "description": response_model.model_json_schema()['description'],
                "parameters": {
                    "type": "object",
                    "properties": response_model.model_json_schema()['properties'],
                    "required": response_model.model_json_schema().get('required', []),
                },
            }
        }]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            print("原始响应:", response)

            arguments = response.choices[0].message.tool_calls[0].function.arguments
            print("解析的参数:", arguments)
            return response_model.model_validate_json(arguments)

        except Exception as e:
            print('ERROR:', e)
            if hasattr(response, 'choices') and len(response.choices) > 0:
                print('Message:', response.choices[0].message)
            return None


# 数据模型定义
class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal[tuple(domain_options)] = Field(description="领域")
    intent: Literal[tuple(intent_options)] = Field(description="意图")
    slots: List[Literal[tuple(slot_options)]] = Field(
        default_factory=list,
        description="实体标签列表"
    )


# API 请求和响应模型
class ExtractionRequest(BaseModel):
    text: str = Field(..., description="需要分析的文本")
    model_name: str = Field(default="qwen-plus", description="使用的模型名称")


class ExtractionResponse(BaseModel):
    success: bool = Field(description="请求是否成功")
    data: Optional[IntentDomainNerTask] = Field(default=None, description="抽取结果")
    error: Optional[str] = Field(default=None, description="错误信息")


# 全局提取器实例
extraction_agent = ExtractionAgent()


# API 路由
@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": "信息抽取服务已启动",
        "version": "1.0.0",
        "available_domains": domain_options,
        "available_intents": intent_options,
        "available_slots": slot_options
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract_intent_domain_ner(request: ExtractionRequest):
    """
    信息抽取接口

    - **text**: 需要分析的文本
    - **model_name**: 使用的模型名称（可选，默认为qwen-plus）
    """
    try:
        # 如果请求中指定了不同的模型，创建新的提取器实例
        if request.model_name != extraction_agent.model_name:
            agent = ExtractionAgent(model_name=request.model_name)
        else:
            agent = extraction_agent

        result = agent.call(request.text, IntentDomainNerTask)

        if result:
            return ExtractionResponse(
                success=True,
                data=result
            )
        else:
            return ExtractionResponse(
                success=False,
                error="信息抽取失败，请检查输入文本或稍后重试"
            )

    except Exception as e:
        return ExtractionResponse(
            success=False,
            error=f"处理请求时发生错误: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "information_extraction"}


# 启动应用
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # 允许外部访问
        port=8066,  # 端口号
        reload=True  # 开发时自动重载
    )
    #uvicorn 作业2_tools_fastapi:app --host 0.0.0.0 --port 8066 --reload。访问http://localhost:8066/docs
