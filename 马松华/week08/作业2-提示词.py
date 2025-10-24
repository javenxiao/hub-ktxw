#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Author: Marky
 Time: 2025/10/19 20:43 
 Description:
 尝试写一下解决意图识别 + 领域识别 + 实体识别的过程。
"""
import openai

client = openai.OpenAI(
    api_key='sk-d2e480f5618947c3a036ed188489bcfb',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

# 提示词 方式
messages = [
    {"role": "system", "content": """假如你是一个信息抽取专家，请对下面的内容抽取它的领域类别，意图类型，实体标签
-- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / stor
-- 待选的意图类型：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITIO
-- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_tim
最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

{
    "domain": ,
    "intent": ,
    "slots": {
      "待选实体": "实体名词",
    }
}
 
    """},
    {"role":"user","content":"重庆的乌江风景很美"}
]

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=messages
)

result = completion.choices[0].message.content
print(result)

# {
#     "domain": "stor",
#     "intent": "DEFAULT",
#     "slots": {
#       "location_city": "重庆",
#       "location_poi": "乌江"
#     }
# }
