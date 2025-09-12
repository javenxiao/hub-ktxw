import requests
import json

url = "http://localhost:11434/v1/completions"
data = {
  "model": "qwen3:0.6b",
  "prompt": "天空为什么是蓝色的？"
}
headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, json=data)
print("Status Code", response.status_code)
print("JSON Response ", response.json())