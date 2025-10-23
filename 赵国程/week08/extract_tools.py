from pydantic import BaseModel, Field
from typing import List, Union, Dict
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-88d3ca9887854f7e92a8485baa49993f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
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


class IntentExtract(BaseModel):
    """分析句子中的意图"""
    intent: Literal['OPEN','SEARCH','REPLAY_ALL','NUMBER_QUERY','DIAL','CLOSEPRICE_QUERY','SEND','LAUNCH','PLAY','REPLY','RISERATE_QUERY','DOWNLOAD','QUERY','LOOK_BACK','CREATE','FORWARD','DATE_QUERY','SENDCONTACTS','DEFAULT','TRANSLATION','VIEW','NaN','ROUTE','POSITION'] = Field(description="意图")

class DomainExtract(BaseModel):
    """分析句子中的领域"""
    domain: Literal["music","app","radio","lottery","stock","novel","weather","match","map","website","news","message","contacts","translation","tvchannel","cinemas","cookbook","joke","riddle","telephone","video","train","poetry","flight","epg","health","email","bus","story"] = Field(description="领域")

class EntitiesExtract(BaseModel):
    """分析句子中的实体及标签"""
    label: Literal['code','Src','startDate_dateOrig','film','endLoc_city','artistRole','location_country','location_area','author','startLoc_city','season','dishNamet','media','datetime_date','episode','teleOperator','questionWord','receiver','ingredient','name','startDate_time','startDate_date','location_province','endLoc_poi','artist','dynasty','area','location_poi','relIssue','Dest','content','keyword','target','startLoc_area','tvchannel','type','song','queryField','awayName','headNum','homeName','decade','payment','popularity','tag','startLoc_poi','date','startLoc_province','endLoc_province','location_city','absIssue','utensil','scoreDescr','dishName','endLoc_area','resolution','yesterday','timeDescr','category','subfocus','theatre','datetime_time'] = Field(description="标签")
    entity: List[Dict[str, str]] = Field("实体, 实体的标签")

def extract_tools(request_text: Union[str, List[str]]) -> Union[Dict,List[Dict]]:
    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")
    results =[]
    for text in request_text:
        intent = ExtractionAgent("qwen-plus").call(text, IntentExtract)
        domain = ExtractionAgent("qwen-plus").call(text, DomainExtract)
        entities = ExtractionAgent("qwen-plus").call(text, EntitiesExtract)
        result = {
            "intent": intent.intent,
            "domain": domain.domain,
            "entities": entities
        }
        results.append(result)
    if len(results) == 1:
        return results[0]
    return results

