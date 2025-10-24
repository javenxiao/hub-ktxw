import openai
import json
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List, Optional, Dict, Any
from typing_extensions import Literal

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-4274c13a17904fff983a8c761c5bec9f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

"""
这个智能体（不是满足agent的功能），能自动生成tools的json，实现信息信息抽取
"""
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
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        # "required": response_model.model_json_schema()['required'], # 必须要传的参数
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
            print('ERROR', response.choices[0].message)
            return None

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather', 'match', 'map', 'website', 'news', 'message',
                    'contacts', 'translation', 'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone', 'video', 'train',
                    'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY',
                    'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY',
                    'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT',
                    'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'] = Field(description="意图")
    slots: Dict[str, Any] = Field(description="实体槽位字典", default_factory=dict)

# 测试代码
if __name__ == '__main__':
    test_cases = [
        "帮我查询下从北京到天津到武汉的汽车票",
        "播放周杰伦的青花瓷",
        "查询上海今天的天气",
        "搜索最近的餐厅",
        "张三叫李四叔叔"
    ]

    agent = ExtractionAgent(model_name="qwen-plus")

    for i, text in enumerate(test_cases, 1):
        print(f"\n测试用例{i}: {text}")
        result = agent.call(text, IntentDomainNerTask)
        if result is not None:
            print(f"领域: {result.domain}")
            print(f"意图: {result.intent}")
            print(f"实体: {json.dumps(result.slots, ensure_ascii=False, indent=2)}")
        else:
            print("抽取失败")

