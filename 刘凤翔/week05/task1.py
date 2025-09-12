import openai

client = openai.OpenAI(
    api_key="sk-bd00d5a43fd8498bb4383159a88c2a98",
    base_url="https://api.deepseek.com/v1"
)

for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "今天发生了什么重大新闻？"}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
