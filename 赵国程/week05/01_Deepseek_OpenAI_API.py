import openai

client = openai.OpenAI(
    api_key="sk-9327b8540d************",
    base_url="https://api.deepseek.com"
)

for resp in client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "你是一个专业的数学老师"},
            {"role": "user", "content": "证明费马大定理"}
        ],
        stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
