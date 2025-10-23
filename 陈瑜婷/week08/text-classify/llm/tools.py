import openai
import json
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List, Optional, Dict
from typing_extensions import Literal
import os

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def classify_by_tools(request_text: str) -> str:
    result = ExtractionAgent(model_name = "qwen-plus").call(request_text, IntentDomainNerTask)
    return result

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
    domain: Literal['music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather', 'match', 'map', 'website', 'news', 'message', 'contacts', 'translation', 'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone', 'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY', 'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY', 'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT', 'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'] = Field(description="意图")
    entity: Dict[Literal['code', 'Src', 'startDate_dateOrig', 'film', 'endLoc_city', 'artistRole', 'location_country', 'location_area', 'author', 'startLoc_city', 'season', 'dishNamet', 'media', 'datetime_date', 'episode', 'teleOperator', 'questionWord', 'receiver', 'ingredient', 'name', 'startDate_time', 'startDate_date', 'location_province', 'endLoc_poi', 'artist', 'dynasty', 'area', 'location_poi', 'relIssue', 'Dest', 'content', 'keyword', 'target', 'startLoc_area', 'tvchannel', 'type', 'song', 'queryField', 'awayName', 'headNum', 'homeName', 'decade', 'payment', 'popularity', 'tag', 'startLoc_poi', 'date', 'startLoc_province', 'endLoc_province', 'location_city', 'absIssue', 'utensil', 'scoreDescr', 'dishName', 'endLoc_area', 'resolution', 'yesterday', 'timeDescr', 'category', 'subfocus', 'theatre', 'datetime_time'], str] = Field(description="实体")
    # Src: Optional[str] = Field(description="出发地")
    # Des: List[str] = Field(description="目的地")


#print(classify_by_tools("帮我查询下从北京到天津到武汉的汽车票"))
