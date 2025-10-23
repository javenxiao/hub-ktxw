# 第三方库
import openai
from fastapi import FastAPI
from logger import logger

app = FastAPI()

@app.post("/v1/Int_Dom_Ent_rec/prompt")
def Prompt_rec(req) :
    '''用PROMPT进行意图识别&领域识别&实体识别'''
    try:
        client = openai.OpenAI(
            api_key="sk-67a5c9ec491c4832b45b0cbb567bc67f",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content":
                    f"""对下面的文本进行三种大的识别，识别其中的细分类别：
                    domain_recognition：music/app/radio/lottery/stock/novel/weather/match/map/website/news/message/contacts/translation/tvchannel/cinemas/cookbook/joke/riddle/telephone/video/train/poetry/flight/epg/health/email/bus/story
                    intent_recognition：OPEN/SEARCH/REPLAY_ALL/NUMBER_QUERY/DIAL/CLOSEPRICE_QUERY/SEND/LAUNCH/PLAY/REPLY/RISERATE_QUERY/DOWNLOAD/QUERY/LOOK_BACK/CREATE/FORWARD/DATE_QUERY/SENDCONTACTS/DEFAULT/TRANSLATION/VIEW/NaN/ROUTE/POSITION
                    entity_recognition：code/Src/startDate_dateOrig/film/endLoc_city/artistRole/location_country/location_area/author/startLoc_city/season/dishNamet/media/datetime_date/episode/teleOperator/questionWord/receiver/ingredient/name/startDate_time/startDate_date/location_province/endLoc_poi/artist/dynasty/area/location_poi/relIssue/Dest/content/keyword/target/startLoc_area/tvchannel/type/song/queryField/awayName/headNum/homeName/decade/payment/popularity/tag/startLoc_poi/date/startLoc_province/endLoc_province/location_city/absIssue/utensil/scoreDescr/dishName/endLoc_area/resolution/yesterday/timeDescr/category/subfocus/theatre/datetime_time
                    输出结果内容格式，比如：
                    "intent_recognition":"OPEN","SEARCH"
                    "domain_recognition":"music","app"
                    "intent_recognition":"OPEN","SEARCH"
                    输出要求：除了输出的内容，不输出其他的。
                    文本：{req}
                    """},
            ],
        )
        # 需要将返回内容解析并包装成 TextClassifyResponse 对象
        # 这里需要根据实际的 TextClassifyResponse 定义来实现
        content=completion.choices[0].message.content
        return content
    except Exception as e:
        logger.error(f"Prompt classify error: {str(e)}")
        raise

@app.post("/v1/Int_Dom_Ent_rec/Tools")
def Tools_rec(req) :
    '''用Tools进行意图识别&领域识别&实体识别'''
    from Tools.ExtractionAgent import ExtractionAgent
    from Tools.Text import Text
    result = ExtractionAgent(model_name="qwen-plus").call(req, Text)
    return result





