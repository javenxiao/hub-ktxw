import base64
import requests
import os
from openai import OpenAI


# 配置云端 qwen3-vl 模型接口
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-0fce7eb19bc042469a27ea5005b65635"  # 请将 API Key 配置到环境变量中

# 初始化OpenAI客户端
client = OpenAI(api_key=API_KEY, base_url=API_URL)


def image_to_base64(image_path):
    """将本地图片转为 base64 编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



def ocr_img(image_path):
    """调用云端 qwen3-vl-plus 模型识别图片中的文字"""
    base64_image = image_to_base64(image_path)
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
                            "text": "请判断这张图片中的文字,并提取出所有的文字。"
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

def main():
    image_path = "week10/1.png"
    ocr_result = ocr_img(image_path)
    print(ocr_result)

if __name__ == "__main__":
    main()
