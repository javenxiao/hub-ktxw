import openai

client = openai.OpenAI(
    api_key="sk-5f2c2***********************9cec",
    base_url="https://api.deepseek.com/v1"
)

for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "证明费马大定理"},
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
