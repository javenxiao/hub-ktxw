import openai
from pydantic import BaseModel, Field
from typing_extensions import Literal
from data_schema import TextResponse

def queryBytools(request_text: str) -> TextResponse:
    client = openai.OpenAI(
        api_key="sk-130325d22d86441db5f7d18866cd4dce",  # https://bailian.console.aliyun.com/?tab=model#/api-key
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

    class text(BaseModel):
        """根据用户提供的信息抽取它的意图类型、领域类别、实体标签"""
        intent: Literal[
            "OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"] = Field(
            description="意图类型")
        domain: Literal[
            "music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "story"] = Field(
            description="领域类别")
        slots: Literal[
            "code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole", "location_country", "location_area", "author", "startLoc_city", "season", "dishNamet", "media", "datetime_date", "episode", "teleOperator", "questionWord", "receiver", "ingredient", "name", "startDate_time", "startDate_date", "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi", "relIssue", "Dest", "content", "keyword", "target", "startLoc_area", "tvchannel", "type", "song", "queryField", "awayName", "headNum", "homeName", "decade", "payment", "popularity", "tag", "startLoc_poi", "date", "startLoc_province", "endLoc_province", "location_city", "absIssue", "utensil", "scoreDescr", "dishName", "endLoc_area", "resolution", "yesterday", "timeDescr", "category", "subfocus", "theatre", "datetime_time"] = Field(
            description="实体标签")
    result = ExtractionAgent(model_name="qwen-plus").call(request_text, text)
    return result