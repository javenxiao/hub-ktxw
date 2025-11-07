from http import HTTPStatus
import dashscope
import base64
import os

def extract_text_from_screenshot(image_path_or_url, is_local_file=False):
    """
    使用Qwen-VL模型解析带文字截图的图像并提取文本

    Args:
        image_path_or_url: 图像路径或URL
        is_local_file: 是否为本地文件，默认为False（URL）

    Returns:
        提取的文本内容
    """
    # 准备图像内容
    if is_local_file:
        # 本地文件转换为base64
        with open(image_path_or_url, "rb") as f:
            image_content = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    else:
        # 直接使用URL
        image_content = image_path_or_url

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_content},
                {"text": "请提取这张图片中的所有文字内容，并按照原文格式输出。如果图片中没有文字，请回复'未检测到文字'。"}
            ]
        }
    ]

    try:
        # 调用通义千问VL模型
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-plus',  # 或 'qwen-vl-max'
            messages=messages
        )

        # 检查响应状态
        if response.status_code == HTTPStatus.OK:
            # 提取模型返回的文本内容
            content_list = response.output.choices[0].message.content
            extracted_text = "".join([item.get('text', '') for item in content_list]).strip()

            return extracted_text
        else:
            return f"API请求失败: {response.code} - {response.message}"

    except Exception as e:
        return f"发生异常: {str(e)}"


# 使用示例
if __name__ == "__main__":
    # 设置API密钥
    dashscope.api_key = "sk-4274c13a17904fff983a8c761c5bec9f"  # 替换为您的实际API密钥

    print("=== 单张截图文字提取 ===")
    screenshot_url = "https://pic.rmb.bdstatic.com/bjh/events/3df249d26fa04335100e9de2d8dd8abf2524.jpeg@h_1280"
    text_result = extract_text_from_screenshot(screenshot_url)
    print(f"提取的文字: {text_result}")
