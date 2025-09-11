import requests

url = 'http://192.168.199.75:8000/api/generate'
data = {
    "model": "qwen3:0.6b",
    "prompt": "为什么天空是蓝色的？",
    "stream": False
}

response = requests.post(url, json=data)
response_json = response.json()

print(response_json['response'])
