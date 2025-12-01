from openai import OpenAI
import base64

client = OpenAI(
    api_key="sk-965710e5b13541a493c47ffa1c8c4634",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 图片base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 故事
base64_image = encode_image("text.png")

completion = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[
        {
            "role": "system",
            "content": "你是一个图片理解大师，擅长提取图片的文字。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
                {"type": "text", "text": "请提取图片中的文字"},
            ]
        },
    ],
)
print(f"图片的文字：{completion.choices[0].message.content}")