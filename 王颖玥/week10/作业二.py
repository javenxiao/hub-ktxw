from openai import OpenAI
import base64
import os

client = OpenAI(
    api_key="sk-b8f16de2371547c397349552ecffce68",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

image_path = ["./screenshot/1.jpg", "./screenshot/2.jpg"]


def image_to_base64_url(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
    base64_str = base64.b64encode(image_data).decode("utf-8")
    ext = os.path.splitext(image_path)[1].replace(".", "")
    return f"data:image/{ext};base64,{base64_str}"


def extract_text(image_url):
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
                            "text": "请从这张图中提取出所有文字"
                        }
                    ]
                }
            ],
            max_tokens=512
        )

        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        return f"调用失败: {str(e)}"


for i, each in enumerate(image_path):
    image_url = image_to_base64_url(each)
    result = extract_text(image_url)
    print(f"第{i + 1}张截图：")
    print(f"{result}")


