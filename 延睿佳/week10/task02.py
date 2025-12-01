from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_image = encode_image("E:/study/AI/image/screenshot.png")

client = OpenAI(
    api_key="sk-79522777ae4e4564857b513bf950825c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-vl",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
                {
                    "type": "text",
                    "text": "请识别图片中的所有文字，并输出为纯文本（不要描述图片内容）。",
                },
            ],
        }
    ],
    temperature=0.0,
)
print(completion.choices[0].message.content)
