from openai import OpenAI
from pydantic import BaseModel,Field
from typing import List,Optional,Union
from typing_extensions import Literal
class Extract:
    def __init__(self,model_name):
        self.model_name = model_name
    def call(self,query_text,response_model):
        client = OpenAI(api_key="ollama",
                        base_url="http://localhost:8000/v1")
        tools = [
            {
                "type":"function",
                "function":{
                    "name":response_model.model_json_schema()['title'],
                    "description":response_model.model_json_schema()['description'],
                    "parameters":{
                        "type":"object",
                        "properties":response_model.model_json_schema()['properties']
                    },
                    "required":response_model.model_json_schema()['required']
                }
            }
        ]
        messages = [{"role":"system","content":"你是一个专业的信息抽取专家，帮我抽取下列文本的领域，意图和实体标签"},
                    {"role":"user","content":query_text}]
        response = client.chat.completions.create(messages=messages,
                                                model=self.model_name,
                                                tools=tools,
                                                tool_choice='auto')
        result = response.choices[0].message
        try:
            arguments = result.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None
class DomainIntentEntity(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    Domain:Literal['music', 'app', 'weather', 'bus','flight'] = Field(description="领域")
    Intent:Literal['OPEN', 'SEARCH', 'QUERY','PLAY'] = Field(description="意图")
    # Src: Optional[str] = Field(description="出发地")
    # Des: Union[str,list[str]] = Field(description="目的地")
    artist:Optional[str] = Field(description="艺术家")
    song : Optional[str] = Field(description="歌名")
# print(Extract("qwen3:0.6b").call("帮我播放周杰伦的歌曲",DomainIntentEntity))
def tools_llm(request_text,model_name):
    if isinstance(request_text,str):
        request_text = [request_text]
    elif isinstance(request_text,list):
        pass
    else:
        raise Exception("格式不支持")
    results = []
    for text in request_text:
        agent = Extract(model_name)
        result = agent.call(text,DomainIntentEntity)
        if result is not None:
            results.append(result.model_dump())
        else:
            results.append({"result":"无法识别"})
    return results
