from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-67a5c9ec491c4832b45b0cbb567bc67f", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


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
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
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


class Text(BaseModel):
    """进行意图识别"""
    intent_recognition: List[Literal["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"]] = Field(description="根据输入的语句，进行意图识别")
    domain_recognition: List[Literal["music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "story"]] = Field(description="根据输入的语句，进行领域识别")
    entity_recognition: List[Literal["code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole", "location_country", "location_area", "author", "startLoc_city", "season", "dishNamet", "media", "datetime_date", "episode", "teleOperator", "questionWord", "receiver", "ingredient", "name", "startDate_time", "startDate_date", "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi"]]= Field(description="根据输入的语句，进行实体识别")
result = ExtractionAgent(model_name = "qwen-plus").call('把高建清的号码发给潘青华', Text)
print(result)

