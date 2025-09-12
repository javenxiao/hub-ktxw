
from openai import OpenAI

client = OpenAI(api_key="sk-ba6c9*******3c12fcfd2893b3", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat", # 非思考模式
    # model="deepseek-reasoner", # 思考模式
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "我在做一个ai接口测试"},
    ],
    stream=False
)


print(response.choices[0].message.content)
