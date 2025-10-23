from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
def prompt_llm(request_text,model_name):

    llm = ChatOpenAI(api_key="ollama",
                     base_url="http://localhost:8000/v1",
                     model = model_name)
    template = """
    你是一个专业信息抽取专家，请对下面的文本抽取它的领域类别，意图类别，实体标签，所有抽取出的结果都必须在下面对应的类别中，如果没有则输出无匹配结果
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
        "slots": {{
          "待选实体": "实体名词",
        }}
    }}
    ```
    """
    parser = JsonOutputParser()
    prompt_template = ChatPromptTemplate.from_messages([('system',template),('user','{query_text}')])
    if isinstance(request_text,str):
        request_text = [request_text]
    elif isinstance(request_text,list):
        pass
    else:
        raise Exception("格式不支持")
    results = []
    for text in request_text:
        chain = prompt_template | llm | parser
        result = chain.invoke({"query_text":text})
        if isinstance(result,dict):
            results.append(result)
        else:
            results.append({"result":"无法识别"})
    return results
print(prompt_llm(["帮我查询北京到上海的飞机票","帮我播放周杰伦的歌曲"],"deepseek-r1:8b"))
