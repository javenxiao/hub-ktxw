from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal
from fastapi import FastAPI

import openai
import json

client = openai.OpenAI(
    api_key="sk-0fce7eb19bc042469a27ea5005b65635", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 初始化FastAPI应用
app = FastAPI()

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
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
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

# 读取文件内容，把文件中line.strip()后的内容添加到list中
domains_list = []
intents_list = []
slots_list = []


def read_files(file_path, list_name):
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            list_name.append(line.strip())


read_files("data/domains.txt", domains_list)
read_files("data/intents.txt", intents_list)
read_files("data/slots.txt", slots_list)

class ExtractionResponse(BaseModel):
    """领域识别，意图识别与实体抽取结果"""
    domain: Literal[*domains_list] = Field(description="领域类别")
    intent: Literal[*intents_list] = Field(description="意图类别")
    slots: dict[Literal[*slots_list], str] = Field(description="实体类别")

class RequestBody(BaseModel):
    """API请求体"""
    user_prompt: str

# 初始化ExtractionAgent实例
extraction_agent = ExtractionAgent(model_name="qwen-plus")

@app.post("/extract")
async def extract_info(request: RequestBody):
    """
    API接口，用于进行领域识别、意图识别与实体抽取
    """
    result = extraction_agent.call(request.user_prompt, ExtractionResponse)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
