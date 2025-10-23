import json

from pydantic import BaseModel, Field
from typing import Literal, List
from openai import OpenAI


def get_tags(path):
    """
    获取类型、标签
    :param path: 文件名
    :return: 类型、标签字符串
    """
    tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tags.append(line.strip())
    return tags


intents = get_tags('intents.txt')
# print(intents)
domains = get_tags('domains.txt')
# print(domains)
slots = get_tags('slots.txt')
# print(slots)


class RecognitionModel(BaseModel):
    """
    识别数据类
    """
    intent: Literal[*intents] = Field(..., description='意图类型')
    domain: Literal[*domains] = Field(..., description='领域类型')
    slot: List[tuple[Literal[*slots], str]] = Field(..., description='实体标签和对应的实体名词')


client = OpenAI(
    api_key='sk-965710e5b13541a493c47ffa1c8c4634',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)


def recognition(content):
    """
    意图识别、领域识别、实体识别
    :param content: 输入内容
    :return: 输出结果 json格式
    """
    messages = [
        {
            'role': 'user',
            'content': content
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": RecognitionModel.model_json_schema()['title'],  # 工具名字
                "description": RecognitionModel.model_json_schema()['description'],  # 工具描述
                "parameters": {
                    "type": "object",
                    "properties": RecognitionModel.model_json_schema()['properties'],  # 参数说明
                    "required": RecognitionModel.model_json_schema()['required'], # 必须要传的参数
                },
            }
        }
    ]
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
    )
    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls and len(tool_calls) > 0:
        tool_call = tool_calls[0]
        arguments = tool_call.function.arguments
        try:
            data = json.loads(arguments)
            slot = dict()
            for item in data['slot']:
                slot[item[0]] = item[1]
            result = {
                'domain': data['domain'],
                'intent': data['intent'],
                'slot': slot,
            }
            print(result)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            print(e)
    return None


# 测试
# if __name__ == '__main__':
#     result = recognition('糖醋鲤鱼怎么做啊？你只负责吃，c则c。')
#     print('识别结果：\n', result)
#     result = recognition('从台州到金华的汽车票。')
#     print('识别结果：\n', result)
