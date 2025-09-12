from openai import OpenAI

client = OpenAI(
    api_key='sk-ae088c717d*****19207effc4c',
    base_url="https://api.deepseek.com"
)

messages = [
    {
        "role": "system",
        "content": "你是一个经验丰富的起名大师，擅长易经、五行等，对诗词也有深入的研究"
    },
    {
        "role": "user",
        "content": "今年是2026年3月份或者4月份，请帮忙起3个姓冯，立字辈的名字，要求名字郎朗上口，有诗意，分数较高"
    }
]

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)

content = response.choices[0].message.content


print(content)
