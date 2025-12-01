from openai import OpenAI
import os
import base64

# 初始化OpenAI客户端
client = OpenAI(
    api_key = "sk-88d3ca9887854f7e92a8485baa49993f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

answer_content = ""     # 定义完整回复

#  编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 编码图片
base64_image = encode_image("task10/img/3.png")

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
                {"type": "text", "text": "请帮我提取图片中的文字内容，不需要其他解释。"},
            ],
        },
    ],
    stream=True,
    # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
    extra_body={
        'enable_thinking': False,
        "thinking_budget": 81920},
)


for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 开始回复
        if delta.content != "" and is_answering is False:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
            is_answering = True
        # 打印回复过程
        print(delta.content, end='', flush=True)
        answer_content += delta.content
