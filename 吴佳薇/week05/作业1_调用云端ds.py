from openai import OpenAI

client=OpenAI(api_key="sk-eaa872cf1aeb42cd842c3d977d2e35df",base_url="https://api.deepseek.com/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)

#已申请api_key，但还未充值
