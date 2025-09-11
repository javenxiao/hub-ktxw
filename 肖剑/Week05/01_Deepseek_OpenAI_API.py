# 申请deepseek的api，https://platform.deepseek.com/usage， 使用openai 库调用云端大模型。
from openai import OpenAI

client = OpenAI(api_key="sk-a232d10*****4187ac2a6eb7b3e*****", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",  # deepseek-chat 对应 DeepSeek-V3.1 的非思考模式，deepseek-reasoner 对应 DeepSeek-V3.1 的思考模式
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "武术和功夫的区别？"},
    ],
    stream=True  # 流式
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
