from openai import OpenAI
import base64

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-88d3ca9887854f7e92a8485baa49993f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 编码图片
base64_image = encode_image("/Users/zhaoguocheng/PycharmProjects/nlp_course/task10/img/gou.jpeg")

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
                {"type": "text", "text": "请分析这张图片，判断它是狗还是猫。只需要回答狗或猫，不需要其他解释。"},
            ],
        },
    ],
    stream=False,
    # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
    extra_body={
        'enable_thinking': True,
        "thinking_budget": 81920},
)

print(completion.choices[0].message.content.strip())
