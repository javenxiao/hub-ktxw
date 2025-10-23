import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import json


def query(request_text: str) -> str:
    os.environ["OPENAI_API_KEY"] = "sk-130325d22d86441db5f7d18866cd4dce"
    os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    model = init_chat_model("qwen-plus", model_provider="openai")

    system_template = """你是一个专业的信息抽取专家，请对下面的文本抽取它的意图类型、领域类别、实体标签：
            - 待选的意图类别(intent)：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
            - 待选的领域类别(domain)：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
            - 待选的实体标签(slots)：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time
            请输出严格的JSON格式：{{\"intent\": \"意图\", \"domain\": \"领域\", \"slots\": \"实体\"}}
            """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"text": request_text})
    # 调用模型
    response = model.invoke(prompt)
    return response
