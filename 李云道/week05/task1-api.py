from openai import OpenAI

'''
申请deepseek的api，https://platform.deepseek.com/usage， 使用openai 库调用云端大模型。
'''

client = OpenAI(api_key="sk-7e9963be*******19655", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你是谁"},
    ],
    stream=False
)

print(response.choices[0].message.content)
"""
我是DeepSeek-V3，由深度求索公司打造的智能助手！😊 我的目标是帮助你解答各类问题，无论是学习、工作，还是生活中的小困扰，我都会尽力提供有用的信息和建议。有什么想问的，尽管告诉我吧！
"""
