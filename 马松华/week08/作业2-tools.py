#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Author: Marky
 Time: 2025/10/20 22:59 
 Description:
"""
import openai

client = openai.OpenAI(
    api_key='sk-d2e480f5618947c3a036ed188489bcfb',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_information",
            "description": "抽取领域类别，意图类型，实体标签",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "description": "领域类别",
                        "type": "string",
                        "enum": ["music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "stor"],
                    },
                    "intent": {
                        "description": "意图类型",
                        "type": "string",
                        "enum": ["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITIO"],

                    },
                    "slots": {
                        "description": "实体标签",
                        "type": "object",
                        "properties":{
                            "code": {"type": "string"},
                            "Src": {"type": "string"},
                            "startDate_dateOrig": {"type": "string"},
                            "film": {"type": "string"},
                            "endLoc_city": {"type": "string"},
                            "artistRole": {"type": "string"},
                            "location_country": {"type": "string"},
                            "location_area": {"type": "string"},
                            "author": {"type": "string"},
                            "startLoc_city": {"type": "string"},
                            "season": {"type": "string"},
                            "dishNamet": {"type": "string"},
                            "media": {"type": "string"},
                            "datetime_date": {"type": "string"},
                            "episode": {"type": "string"},
                            "teleOperator": {"type": "string"},
                            "questionWord": {"type": "string"},
                            "receiver": {"type": "string"},
                            "ingredient": {"type": "string"},
                            "name": {"type": "string"},
                            "startDate_time": {"type": "string"},
                            "startDate_date": {"type": "string"},
                            "location_province": {"type": "string"},
                            "endLoc_poi": {"type": "string"},
                            "artist": {"type": "string"},
                            "dynasty": {"type": "string"},
                            "area": {"type": "string"},
                            "location_poi": {"type": "string"},
                            "relIssue": {"type": "string"},
                            "Dest": {"type": "string"},
                            "content": {"type": "string"},
                            "keyword": {"type": "string"},
                            "target": {"type": "string"},
                            "startLoc_area": {"type": "string"},
                            "tvchannel": {"type": "string"},
                            "type": {"type": "string"},
                            "song": {"type": "string"},
                            "queryField": {"type": "string"},
                            "awayName": {"type": "string"},
                            "headNum": {"type": "string"},
                            "homeName": {"type": "string"},
                            "decade": {"type": "string"},
                            "payment": {"type": "string"},
                            "popularity": {"type": "string"},
                            "tag": {"type": "string"},
                            "startLoc_poi": {"type": "string"},
                            "date": {"type": "string"},
                            "startLoc_province": {"type": "string"},
                            "endLoc_province": {"type": "string"},
                            "location_city": {"type": "string"},
                            "absIssue": {"type": "string"},
                            "utensil": {"type": "string"},
                            "scoreDescr": {"type": "string"},
                            "dishName": {"type": "string"},
                            "endLoc_area": {"type": "string"},
                            "resolution": {"type": "string"},
                            "yesterday": {"type": "string"},
                            "timeDescr": {"type": "string"},
                            "category": {"type": "string"},
                            "subfocus": {"type": "string"},
                            "theatre": {"type": "string"},
                            "datetime_tim": {"type": "string"}
                        },
                        "additionalProperties": False
                    },
                },
                "required": ["domain", "intent", "slots"],
                "additionalProperties": False
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "重庆的乌江风景很美"
    }
]


response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

result = response.choices[0].message.tool_calls[0].function.arguments
print(result)
# {"domain": "stor", "intent": "DEFAULT", "slots": {"location_city": "重庆", "keyword": "乌江风景"}}



