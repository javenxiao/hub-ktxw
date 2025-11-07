from http import HTTPStatus
import dashscope

def classify_dog_cat():
    """
    使用Qwen-VL模型对图像进行分类，判断是狗还是猫
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": "https://b0.bdstatic.com/3ba91bdd3a4077c1ca5145620a3455b4.jpg"},  # 猫：https://b0.bdstatic.com/3ba91bdd3a4077c1ca5145620a3455b4.jpg 狗： https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
                {"text": "这是狗还是猫？回答dog或cat，如果无法确定，请回答'unknown'"}
            ]
        }
    ]

    try:
        # 调用通义千问VL模型[citation:4][citation:8]
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-plus',  # 或 'qwen-vl-max'
            messages=messages
        )

        # 检查响应状态
        if response.status_code == HTTPStatus.OK:
            # 提取模型返回的文本内容
            # 注意：返回内容是一个列表，需要提取所有text字段并拼接[citation:4]
            content_list = response.output.choices[0].message.content
            full_response = "".join([item.get('text', '') for item in content_list]).strip().lower()

            # 根据关键词判断分类
            if 'dog' in full_response:
                return 'dog'
            elif 'cat' in full_response:
                return 'cat'
            else:
                return f"unknown (模型回答: {full_response})"
        else:
            # 处理请求失败的情况
            return f"API请求失败: {response.code} - {response.message}"

    except Exception as e:
        return f"发生异常: {str(e)}"


if __name__ == "__main__":
    result = classify_dog_cat()
    print(f"分类结果: {result}")