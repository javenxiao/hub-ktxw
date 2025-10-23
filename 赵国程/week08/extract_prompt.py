import os
from typing import Union, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = "sk-88d3ca9887854f7e92a8485baa49993f"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = init_chat_model("qwen-plus", model_provider="openai")
system_template = """
你是一个专业的文本提取器，你的任务是从给定的文本中提取出所有领域类别 意图类别 实体标签。
待划分的领域类别：music app radio lottery stock novel weather match map website news message contacts translation
               tvchannel cinemas cookbook joke riddle telephone video train poetry flight epg health
               email bus story
待划分的意图类别：OPEN SEARCH REPLAY_ALL NUMBER_QUERY DIAL CLOSEPRICE_QUERY SEND LAUNCH PLAY REPLY RISERATE_QUERY DOWNLOAD QUERY LOOK_BACK CREATE FORWARD DATE_QUERY SENDCONTACTS DEFAULT TRANSLATION VIEW NaN ROUTE POSITION
待划分的实体标签：code Src startDate_dateOrig film endLoc_city artistRole location_country location_area author startLoc_city season dishNamet media datetime_date episode teleOperator questionWord receiver ingredient name startDate_time startDate_date
最终输出格式填充下面的json， domain 是 领域标签,按空格划分， intent 是 意图标签，按空格划分，slots 是实体识别结果和标签，按空格划分。不需要自己生成其他多余的类别和标签
    ```json
    {{
        "domain": ,
        "intent": ,
        "entities": {{
          "实体": "标签"
        }}
    }}
    ```
"""
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
chain = prompt_template | model | JsonOutputParser()


def extract_prompt(text: Union[str, list]) -> Union[Dict, List]:
    if isinstance(text, str):
        text = [text]
    results = []
    for text in text:
        result = chain.invoke({"text": text})
        results.append(result)
    if len(results) == 1:
        return results[0]
    return results
