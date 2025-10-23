import openai
import json
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List, Optional
from typing_extensions import Literal

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-a2aa801ac63a4f0c949dc133a3614786",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

"""
这个智能体（不是满足agent的功能），能自动生成tools的json，实现信息信息抽取
"""
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        # "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'weather', 'bus'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'QUERY'] = Field(description="意图")
    Src: Optional[str] = Field(description="出发地")
    Des: List[str] = Field(description="目的地")

result = ExtractionAgent(model_name = "qwen-plus").call("帮我查询下从北京到天津到武汉的汽车票", IntentDomainNerTask)
print(result)

