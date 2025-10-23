import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing_extensions import Literal
from fastapi import FastAPI, HTTPException
import uvicorn

# FastAPI应用
app = FastAPI(
    title="语义解析API",
    description="智能对话系统语义解析服务",
    version="1.0.0"
)

# 请求模型
class TextInput(BaseModel):
    text: str = Field(..., description="待解析的自然语言文本")

# 响应模型
class ExtractionResult(BaseModel):
    domain: str = Field(..., description="领域类别")
    intent: str = Field(..., description="意图类型")
    slots: Dict[str, Any] = Field(..., description="实体槽位")

# 语义解析任务模型
class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal[
        'music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather', 
        'match', 'map', 'website', 'news', 'message', 'contacts', 'translation', 
        'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone', 
        'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'
    ] = Field(description="领域")
    intent: Literal[
        'OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY',
        'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY',
        'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT',
        'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'
    ] = Field(description="意图")
    slots: Dict[str, Any] = Field(default_factory=dict, description="实体槽位")

# 初始化OpenAI客户端
def get_openai_client():
    return openai.OpenAI(
        api_key="sk-78cc4e9ac8f44efdb207b7232ed8",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

# 提取智能体
class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
        self.client = get_openai_client()

    def extract_with_prompt(self, text: str) -> Optional[ExtractionResult]:
        """使用提示词方法进行信息抽取"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签

待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story

待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION

待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

请严格按照以下JSON格式输出结果：
```json
{
    "domain": "领域标签",
    "intent": "意图标签",
    "slots": {
        "实体标签": "实体值"
    }
}
```"""
                    },
                    {
                        "role": "user", 
                        "content": text
                    }
                ],
            )
            
            result_text = completion.choices[0].message.content
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result_dict = json.loads(json_str)
                return ExtractionResult(**result_dict)
            return None
            
        except Exception as e:
            print(f"提示词方法错误: {e}")
            return None

    def extract_with_tools(self, text: str) -> Optional[IntentDomainNerTask]:
        """使用Tools方法进行信息抽取"""
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": IntentDomainNerTask.model_json_schema()['title'],
                        "description": IntentDomainNerTask.model_json_schema()['description'],
                        "parameters": {
                            "type": "object",
                            "properties": IntentDomainNerTask.model_json_schema()['properties'],
                        },
                    }
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
                tools=tools,
                tool_choice="auto",
            )
            
            if response.choices[0].message.tool_calls:
                arguments = response.choices[0].message.tool_calls[0].function.arguments
                return IntentDomainNerTask.model_validate_json(arguments)
            return None
            
        except Exception as e:
            print(f"Tools方法错误: {e}")
            return None

# 全局智能体实例
extraction_agent = ExtractionAgent()

# API路由
@app.get("/")
async def root():
    return {"message": "语义解析API服务已启动", "status": "running"}

@app.post("/extract/prompt", response_model=ExtractionResult)
async def extract_with_prompt(input_data: TextInput):
    """使用提示词方法进行语义解析"""
    result = extraction_agent.extract_with_prompt(input_data.text)
    if result:
        return result
    raise HTTPException(status_code=500, detail="语义解析失败")

@app.post("/extract/tools", response_model=ExtractionResult)
async def extract_with_tools(input_data: TextInput):
    """使用Tools方法进行语义解析"""
    result = extraction_agent.extract_with_tools(input_data.text)
    if result:
        return ExtractionResult(
            domain=result.domain,
            intent=result.intent,
            slots=result.slots
        )
    raise HTTPException(status_code=500, detail="语义解析失败")

@app.post("/extract/both")
async def extract_both_methods(input_data: TextInput):
    """同时使用两种方法进行解析并比较结果"""
    prompt_result = extraction_agent.extract_with_prompt(input_data.text)
    tools_result = extraction_agent.extract_with_tools(input_data.text)
    
    return {
        "input_text": input_data.text,
        "prompt_method": prompt_result.dict() if prompt_result else None,
        "tools_method": {
            "domain": tools_result.domain if tools_result else None,
            "intent": tools_result.intent if tools_result else None,
            "slots": tools_result.slots if tools_result else None
        } if tools_result else None
    }

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "semantic_parser"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )
