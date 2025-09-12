import requests
from openai import OpenAI

'''
使用ollama本地模型
'''
BASE_URL = "http://localhost:11434"
client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="ollama")


def process(model: str):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "你是谁"},
        ],
        stream=False
    )
    return response


try:
    response = requests.get(f"{BASE_URL}")
    assert response.text == "Ollama is running"
    model_list = client.models.list()
    print(model_list)
    model_list = [m.id for m in model_list.data]
except:
    print("请先启动ollama服务")
    exit(1)

for model in model_list:
    print(f"Model: {model}")
    response = process(model)
    # for chunk in response:
    #     print(chunk.choices[0].delta.get("content", ""), end="", flush=True)
    print(response.choices[0].message.content)
    print("\n" + "=" * 50 + "\n")
