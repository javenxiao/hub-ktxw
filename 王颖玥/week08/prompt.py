import openai
import json

client = openai.OpenAI(
    api_key="sk-b8f16de2371547c397349552ecffce68",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

text_list = ["查询兰州到广州的飞机。",
             "给我播放张敬轩的歌曲《春秋》。",
             "明天下午4点提醒我找黄婉仪玩。",
             "给我放一部韩剧《请回答1988》。"]

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


results = completion.choices[0].message.content.strip()
print(results)

# {
#     "text": "查询许昌到中山的汽车。",
#     "domain": "bus",
#     "intent": "QUERY",
#     "slots": {
#         "startLoc_city": "许昌",
#         "endLoc_city": "中山"
#     }
# }


