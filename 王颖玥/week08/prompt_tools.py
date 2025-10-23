import openai
import json
from pydantic import BaseModel, Field
from typing import List, Union
from typing_extensions import Literal


client = openai.OpenAI(
    api_key="sk-b8f16de2371547c397349552ecffce68",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# -------提示词抽取---------
def prompt_Extraction(text_list: Union[str, List[str]]):
    texts = "\n".join(text for text in text_list)
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本分别抽取领域类别、意图类别和实体标签
             待选的领域类别有：music/app/radio/lottery/stock/novel/weather/match/map/website/news/message/contacts/translation/tvchannel/cinemas/cookbook/joke/riddle/telephone/video/train/poetry/flight/epg/health/email/bus/story
             待选的意图类别有：OPEN/SEARCH/REPLAY_ALL/NUMBER_QUERY/DIAL/CLOSEPRICE_QUERY/SEND/LAUNCH/PLAY/REPLY/RISERATE_QUERY/DOWNLOAD/QUERY/LOOK_BACK/CREATE/FORWARD/DATE_QUERY/SENDCONTACTS/DEFAULT/TRANSLATION/VIEW/NaN/ROUTE/POSITION
             待选的实体标签有：code/Src/startDate_dateOrig/film/endLoc_city/artistRole/location_country/location_area/author/startLoc_city/season/dishNamet/media/datetime_date/episode/teleOperator/questionWord/receiver/ingredient/name/startDate_time/startDate_date/location_province/endLoc_poi/artist/dynasty/area/location_poi/relIssue/Dest/content/keyword/target/startLoc_area/tvchannel/type/song/queryField/awayName/headNum/homeName/decade/payment/popularity/tag/startLoc_poi/date/startLoc_province/endLoc_province/location_city/absIssue/utensil/scoreDescr/dishName/endLoc_area/resolution/yesterday/timeDescr/category/subfocus/theatre/datetime_time

             要求最终输出的格式是如下的json格式，每一个文本都按照如下格式输出：text是输入文本，domain是领域标签，intent是意图标签，slots是实体识别结果和标签
             {"text": "请帮我打开uc",
                "domain": "app",
                "intent": "LAUNCH",
                "slots": {
                    "待选实体": "uc"
                }
             }
             """},

            {"role": "user", "content": texts},
        ],
    )


    result = completion.choices[0].message.content.strip()
    result = [line.strip() for line in result.splitlines() if line.strip()]
    json_str = "[" + "},{".join("".join(result).split("}{")) + "]"
    result_json = json.loads(json_str)
    return result_json


# -------tools抽取---------
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


class ToolsExtraction(BaseModel):
    """对文本进行抽取领域类别、意图类别和实体标签"""
    domain: Literal["music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "story"] = Field(description="领域类别")
    intent: Literal["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"] = Field(description="意图类别")
    slots: List[EntityAndSlots] = Field(description="实体标签")


def tools_extraction(text_list: Union[str, List[str]]):
    results = []
    if isinstance(text_list, str):
        text_list = [text_list]
    for text in text_list:
        slot_dict = {}
        result = ExtractionAgent(model_name="qwen-plus").call(text, ToolsExtraction)
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

    return results
