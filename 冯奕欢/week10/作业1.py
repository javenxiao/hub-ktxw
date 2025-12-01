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


# 狗
# base64_image = encode_image("dog.jpg")

# 猫
base64_image = encode_image("cat.jpg")

completion = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[
        {
            "role": "system",
            "content": "你是一个图片处理大师，擅长图片分类。"
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
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": "图片里的动物，是猫还是狗？如果是猫，回答猫。如果是狗，回答狗。如果不是猫也不是狗，请回答不知道。不允许编故事"},
            ]
        },
    ],
)
print(f"图片分类：{completion.choices[0].message.content}")