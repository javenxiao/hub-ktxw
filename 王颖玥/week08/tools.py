import json

from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal

import openai

client = openai.OpenAI(
    api_key="sk-b8f16de2371547c397349552ecffce68",
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
                    "parameters": response_model.model_json_schema()
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


class EntityAndSlots(BaseModel):
    """实体及对应标签"""
    label: Literal["code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole", "location_country", "location_area", "author", "startLoc_city", "season", "dishNamet", "media", "datetime_date", "episode", "teleOperator", "questionWord", "receiver", "ingredient", "name", "startDate_time", "startDate_date", "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi", "relIssue", "Dest", "content", "keyword", "target", "startLoc_area", "tvchannel", "type", "song", "queryField", "awayName", "headNum", "homeName", "decade", "payment", "popularity", "tag", "startLoc_poi", "date", "startLoc_province", "endLoc_province", "location_city", "absIssue", "utensil", "scoreDescr", "dishName", "endLoc_area", "resolution", "yesterday", "timeDescr", "category", "subfocus", "theatre", "datetime_time"] = Field(description="实体")
    entity: str = Field(description="标签")


class IntentDomainClassificationAndNerTask(BaseModel):
    """对文本进行抽取领域类别、意图类别和实体标签"""
    domain: Literal["music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "story"] = Field(description="领域类别")
    intent: Literal["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"] = Field(description="意图类别")
    slots: List[EntityAndSlots] = Field(description="实体标签")


text_list = ["查询兰州到广州的飞机。",
             "给我播放张敬轩的歌曲《春秋》。",
             "明天下午4点提醒我找黄婉仪玩。",
             "给我放一部韩剧《请回答1988》。"]

results = []
if isinstance(text_list, str):
    text_list = [text_list]
for text in text_list:
    result = ExtractionAgent(model_name="qwen-plus").call(text, IntentDomainClassificationAndNerTask)
    slot_dict = {}
    for slot in result.slots:
        slot_dict[slot.label] = slot.entity
    if result:
        result_dict = {
            "text": text,
            "domain": result.domain,
            "intent": result.intent,
            "slots": slot_dict
        }
    else:
        result_dict = {text: "抽取失败"}

    results.append(result_dict)

results = json.dumps(results, ensure_ascii=False, indent=2)
print(results)
