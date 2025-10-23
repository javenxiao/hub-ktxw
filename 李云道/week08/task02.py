# openai
import re
from enum import Enum
from typing import Optional, Dict, List

import openai

# langchain
import os

from dns.edns import Option
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# pydantic
from pydantic import BaseModel, Field, field_validator

# project
from config import API_KEY, BASE_URL, USED_MODEL, DOMAINS, INTENTS, SLOTS

'''
使用讲解大模型开发流程（提示词、tools、coze/dify）
尝试写一下解决意图识别 + 领域识别 + 实体识别的过程。
最终效果替代02-joint-bert-training-only(asset文件夹中内容)
可以优先使用coze，不部署dify


领域、意图、主体识别，调教提示词，完善tools的调用
'''

# openai
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
join_pattern = " / "
prompt = f"""你是一名专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 领域类别可选: {join_pattern.join(DOMAINS)}
- 意图类型可选: {join_pattern.join(INTENTS)}
- 待选实体标签: {join_pattern.join(SLOTS)}
最终输出格式填充下面的json，domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签:
```json
{{
    "domain": ,
    "intent": ,
    "slots": {{
        "待选实体标签": "实体名词",
    }}
}}
```
"""

# langchain
os.environ['OPENAI_BASE_URL'] = BASE_URL
os.environ['OPENAI_API_KEY'] = API_KEY
model = init_chat_model(USED_MODEL, model_provider="openai")


# 生成tools调用语法
class ExtractionAgent:
    def __init__(self, model_name: str = USED_MODEL):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        "自动生成tools的json，实现信息信息抽取"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],  # 工具名字
                    "description": response_model.model_json_schema()['description'],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],  # 参数说明
                        "required": response_model.model_json_schema()['required'],  # 必须要传的参数
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
            # 返回response_model的对象
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


# 创建领域枚举
DomainEnum = Enum('DomainEnum', {domain: domain for domain in DOMAINS})

# 创建意图枚举
IntentEnum = Enum('IntentEnum', {intent: intent for intent in INTENTS})

# 创建实体标签枚举（用于验证slots的键）
SlotEnum = Enum('SlotEnum', {slot: slot for slot in SLOTS})


## 定义返回的格式
class NerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Optional[DomainEnum] = Field(..., description="领域类别")
    intent: Optional[IntentEnum] = Field(..., description="意图类型")
    slots: Dict[Optional[SlotEnum], str] = Field(default_factory=dict, description="实体标签列表")


def do_prompt_process_with_openai(question):
    '''使用openai客户端对文本进行领域、意图、实体抽取'''

    completion = client.chat.completions.create(
        model=USED_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    result = completion.choices[0].message.content
    # print(result)
    return result


def do_prompt_process_with_langchain(question):
    '''使用langchain对文本进行领域、意图、实体抽取'''
    messages = [
        ("system", prompt),
        ("user", question)
    ]
    result = model.invoke(messages)
    return result.content


def do_tools_process(question):
    '''使用tools的tools对文本进行领域、意图、实体抽取'''
    result = ExtractionAgent(USED_MODEL).call(question, NerTask)
    # 处理枚举类型的输出
    if result:
        func = lambda x: re.sub(r"<[^:]+: '([^']*)'>", r"'\1'", str(x))
        result = func(result)
    return result


if __name__ == '__main__':
    msg = "糖醋鲤鱼怎么做啊？你只负责吃"

    # result = do_prompt_process_with_openai(msg)
    # print(result)
    #
    # result = do_prompt_process_with_langchain(msg)
    # print(result)

    # tools
    result = ExtractionAgent().call(msg, NerTask)
    print(result)
