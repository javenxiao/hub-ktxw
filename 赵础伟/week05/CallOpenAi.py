from openai import OpenAI


client = OpenAI(
    api_key='sk-41357d21198744d38f05c076348fab3a',
    base_url="https://api.deepseek.com/v1"
)

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
except Exception as e:
    print(f"API调用错误: {str(e)}")