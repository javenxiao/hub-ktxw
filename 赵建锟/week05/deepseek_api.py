import openai

client = openai.OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://api.deepseek.com"
)

for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "证明费马大定理"}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
