from modelscope.models.cv.image_to_image_translation.ops import image_to_base64
from openai import OpenAI
from PIL import Image


client = OpenAI(
    api_key="sk-10f1dde4c437414aa9db4c8142cc6b07",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

image = Image.open(r'./imgs/paper.jpg')

image_base64 = image_to_base64(image)
response = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            },
            {"type": "text", "text": "你是一个专业的图像识别专家。识别图片中的文字并返回给我，只需要图片中的文字"}
        ]
    }]
)
print(response.choices[0].message.content)