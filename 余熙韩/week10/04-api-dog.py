import base64
import requests
import os
from openai import OpenAI


# 配置云端 qwen3-vl 模型接口
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-0fce7eb19bc042469a27ea5005b65635"  # 请将 API Key 配置到环境变量中

# 初始化OpenAI客户端
client = OpenAI(api_key=API_KEY,
                base_url=API_URL)

def image_to_base64(image_path):
    """将本地图片转为 base64 编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def classify_cat_dog(image_path):
    """调用云端 qwen3-vl-plus 模型判断图片是猫还是狗"""
    base64_image = image_to_base64(image_path)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen3-vl-plus",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请判断这张图片是猫还是狗，只需回答“猫”或“狗”。"
                        }
                    ]
                }
            ]
        }
    }

    response = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=payload["input"]["messages"]
    )
    
    answer = response.choices[0].message.content
    return answer.strip()


if __name__ == "__main__":
    # 示例：替换为你的本地图片路径
    img_path = "Week10/0.png"
    try:
        label = classify_cat_dog(img_path)
        print("识别结果：", label)
    except Exception as e:
        print("发生错误：", e)
