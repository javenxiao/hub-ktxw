import openai
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
app = FastAPI()

client = openai.OpenAI(
    api_key=
    "sk-0fce7eb19bc042469a27ea5005b65635",  # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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

output = {"text": "", "domain": "", "intent": "", "slots": {}}

prompt_info = f"""
我需要你扮演一个专业的领域识别、意图识别与实体抽取助手。请严格依据以下提供的类别信息进行识别任务：

### 领域类别
{"/".join(domains_list)}

### 意图类别
{"/".join(intents_list)}

### 实体类别
{"/".join(slots_list)}

输入:
%s

输出格式为json格式,参考如下:
{json.dumps(output, ensure_ascii=False)}
"""

class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: InputText):
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[{
                "role": "user",
                "content": prompt_info % input_data.text
            }],
        )
        return completion.choices[0].message.content
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
