import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal

client = openai.OpenAI(
    api_key="sk-4cb91fbab0154645a837fe4afb5d35ef",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

text_domain = 'music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story'
text_intent = 'OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION'
text_slot = 'code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time'

# 清理数据并创建Literal选项
domain_options = [item.strip() for item in text_domain.split('/') if item.strip()]
intent_options = [item.strip() for item in text_intent.split('/') if item.strip()]
slot_options = [item.strip() for item in text_slot.split('/') if item.strip()]


class ExtractionAgent:
    def __init__(self, model_name: str):
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

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        print("原始响应:", response)

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            print("解析的参数:", arguments)
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print('ERROR:', e)
            print('Message:', response.choices[0].message)
            return None


class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal[tuple(domain_options)] = Field(description="领域")
    intent: Literal[tuple(intent_options)] = Field(description="意图")
    slots: List[Literal[tuple(slot_options)]] = Field(
        default_factory=list,
        description="实体标签列表"
    )


# 测试
result = ExtractionAgent(model_name="qwen-plus").call(
    "帮我查询下从北京到天津到武汉的汽车票",
    IntentDomainNerTask
)
print("最终结果:", result)
