import json
from typing import Dict, Union, List

import openai
from pydantic import BaseModel, Field  # 定义传入的数据请求格式
from typing_extensions import Literal

import config

client = openai.OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
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
        model_json_schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": model_json_schema['title'],
                    "description": model_json_schema['description'],
                    "parameters": {
                        "type": "object",
                        "properties": model_json_schema['properties'],
                        "required": model_json_schema['required'],
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
            print('ERROR:', response.choices[0].message)
            return None


class IntentDomainNerTask(BaseModel):
    """对文本进行意图识别 + 领域识别 + 实体识别"""
    domain: Literal[*config.DOMAINS] = Field(description="领域类别")
    intent: Literal[*config.INTENTS] = Field(description="意图类别")
    slots: Dict[Literal[*config.SLOTS], str] = Field(description="实体类别及实体内容")


def classify_by_tools(request_texts: Union[str, List[str]]) -> Union[dict, List[dict]]:
    if isinstance(request_texts, str):
        return classify(request_texts)

    classify_result = []
    for request_text in request_texts:
        result_dict = classify(request_text)
        classify_result.append(result_dict)

    return classify_result


def classify(request_text: str) -> dict:
    result = ExtractionAgent(config.MODEL).call(request_text, IntentDomainNerTask)
    result_dict = {'text': request_text}
    if result is not None:
        result_dict.update(result)
    else:
        result_dict['error_msg'] = '识别失败'
    return result_dict


if __name__ == "__main__":
    texts = ["帮我查询下桂林飞南京的航班"]
    classify_result = classify_by_tools(texts)
    print(json.dumps(classify_result, ensure_ascii=False, indent=4))
