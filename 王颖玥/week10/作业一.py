from openai import OpenAI
import base64
import os

client = OpenAI(
    api_key="sk-b8f16de2371547c397349552ecffce68",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

image_file = "./image_url.txt"
with open(image_file, "r", encoding="utf-8") as f:
    image_urls = [line.strip() for line in f if line.strip()]


def judge_cat_dog(image_url):
    try:
        response = client.chat.completions.create(
            model='qwen-vl-plus',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "text",
                            "text": "请判断这张图片中的动物是猫还是狗，只回答'猫'，'狗'或'其他'"
                        }
                    ]
                }
            ],
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        return f"调用失败: {str(e)}"


for idx, image_url in enumerate(image_urls):
    result = judge_cat_dog(image_url)
    print(f"图片URL为：{image_url}")
    print(f"识别结果为：{result}")

