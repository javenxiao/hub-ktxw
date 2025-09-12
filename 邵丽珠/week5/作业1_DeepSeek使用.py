import openai
import json


client = openai.OpenAI(
    api_key="******",
    base_url="https://api.deepseek.com/v1",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "今天南京天气怎么样？"},
        {"role": "user", "content": "明天呢？"},
    ],
    stream=False
)

print(response.choices[0].message.content)
