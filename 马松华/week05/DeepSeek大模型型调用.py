import openai

deepseek_api_key = "sk-9f2df94******8c2c6703cd7"
base_url = "https://api.deepseek.com"

client = openai.OpenAI(
    api_key=deepseek_api_key,
    base_url=base_url
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "拼多多当前股价在125美元，市盈率13.5, 从财务，商业模式，市场份额角度来分析当前是否被低估"},
    ],
)
print(completion.model_dump_json())
