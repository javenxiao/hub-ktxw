import openai
import json
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal
from fastapi import FastAPI
from typing import Union


app = FastAPI()

client = openai.OpenAI(
    api_key="sk-2f0f4d94c5d349a3a74830a5e4f4da15",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def model_for_llm(request_text: Union[str, List[str]]) -> List[str]:
    classify_result: Union[str, List[str]]=[]

    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")





    # 大模型提示词
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别，意图标签，实体标签
    -待选的领域类别：music/app/radio/lottery/stock/novel/weather/match/map/website/news/message/contacts/translation/tvchannel/cinemas/cookbook/joke/riddle/telephone/video/train/poetry/flight/epg/health/email/bus/story
    -待选的意图标签：OPEN/SEARCH/REPLAY_ALL/NUMBER_QUERY/DIAL/CLOSEPRICE_QUERY/SEND/LAUNCH/PLAY/REPLY/RISERATE_QUERY/DOWNLOAD/QUERY/LOOK_BACK/CREATE/FORWARD/DATE_QUERY/SENDCONTACTS/DEFAULT/TRANSLATION/VIEW/NaN/ROUTE/POSITION
    -待选的实体标签：code/Src/startDate_dateOrig/film/endLoc_city/artistRole/location_country/location_area/author/startLoc_city/season/dishNamet/media/datetime_date/episode/teleOperator/questionWord/receiver/ingredient/name/startDate_time/startDate_date/location_province/endLoc_poi/artist/dynasty/area/location_poi/relIssue/Dest
    
    最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。
    
    ```json
    {
        "domain": ,
        "intent": ,
        "slots": {
          "待选实体": "实体名词",
        }
    }
    ```
    
   
    """},
            # {"role": "user", "content": [request_text]}
        ],
    )

    classify_result=completion.choices[0].message.content
    return classify_result






# 大模型tools工具
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
    """对文本抽取领域类别、意图类型、实体标签"""
    domain:Literal['music','app','radio','lottery','stock','novel','weather','match','map','website','news','message','contacts','translation','tvchannel','cinemas','cookbook','joke','riddle','telephone','video','train','poetry','flight','epg','health','email','bus','story']=Field(description="领域")
    intents:Literal['OPEN','SEARCH','REPLAY_ALL','NUMBER_QUERY','DIAL','CLOSEPRICE_QUERY','SEND','LAUNCH','PLAY','REPLY','RISERATE_QUERY','DOWNLOAD','QUERY','LOOK_BACK','CREATE','FORWARD','DATE_QUERY','SENDCONTACTS','DEFAULT','TRANSLATION','VIEW','NaN','ROUTE','POSITION']=Field(description="意图")
    slots:Literal['code','Src','startDate_dateOrig','film','endLoc_city','artistRole','location_country','location_area','author','startLoc_city','season','dishNamet','media','datetime_date','episode','teleOperator','questionWord','receiver','ingredient','name','startDate_time','startDate_date']=Field(description="实体")
    # person: List[str] = Field(description="人名")
    # location: List[str] = Field(description="地名")

def model_for_tools(request_text: Union[str, List[str]]) -> List[str]:
    classify_result: Union[str, List[str]]=[]

    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")
    classify_result = ExtractionAgent(model_name = "qwen-plus").call('', Text)


