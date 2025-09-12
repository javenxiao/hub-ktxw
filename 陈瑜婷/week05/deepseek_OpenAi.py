from openai import OpenAI
import os


client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1"
)

messages = [
    {"role": "system", "content": "你是一个AI大模型工程师"},
    {"role": "user", "content": "请简单解释一下大模型的训练过程"}
]

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"API调用错误: {str(e)}")
