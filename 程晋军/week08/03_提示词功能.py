import openai
import json

client = openai.OpenAI(
    api_key="sk-67a5c9ec491c4832b45b0cbb567bc67f", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "user", "content":
            """对下面的文本进行三种大的识别，识别其中的细分类别：
            domain_recognition：music/app/radio/lottery/stock/novel/weather/match/map/website/news/message/contacts/translation/tvchannel/cinemas/cookbook/joke/riddle/telephone/video/train/poetry/flight/epg/health/email/bus/story
            intent_recognition：OPEN/SEARCH/REPLAY_ALL/NUMBER_QUERY/DIAL/CLOSEPRICE_QUERY/SEND/LAUNCH/PLAY/REPLY/RISERATE_QUERY/DOWNLOAD/QUERY/LOOK_BACK/CREATE/FORWARD/DATE_QUERY/SENDCONTACTS/DEFAULT/TRANSLATION/VIEW/NaN/ROUTE/POSITION
            entity_recognition：code/Src/startDate_dateOrig/film/endLoc_city/artistRole/location_country/location_area/author/startLoc_city/season/dishNamet/media/datetime_date/episode/teleOperator/questionWord/receiver/ingredient/name/startDate_time/startDate_date/location_province/endLoc_poi/artist/dynasty/area/location_poi/relIssue/Dest/content/keyword/target/startLoc_area/tvchannel/type/song/queryField/awayName/headNum/homeName/decade/payment/popularity/tag/startLoc_poi/date/startLoc_province/endLoc_province/location_city/absIssue/utensil/scoreDescr/dishName/endLoc_area/resolution/yesterday/timeDescr/category/subfocus/theatre/datetime_time
            输出结果内容格式，比如：
            "intent_recognition":"OPEN","SEARCH"
            "domain_recognition":"music","app"
            "intent_recognition":"OPEN","SEARCH"
            输出要求：除了输出的内容，不输出其他的。
            文本：我想听周杰伦的歌
            """},
    ],
)
print("\nZero-Shot Prompting")
print(completion.choices[0].message.content)



# 其他方法
# https://www.promptingguide.ai/techniques
