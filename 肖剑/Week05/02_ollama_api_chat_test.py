import requests

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "武术和功夫的区别？"}
]

url = 'http://192.168.199.75:8000/api/generate'
data = {
    "model": "qwen3:0.6b",
    "prompt": messages[-1]["content"],
    "stream": False
}

response = requests.post(url, json=data)
response_json = response.json()

print(response_json['response'])