import dashscope
from dashscope import MultiModalConversation
import os
import base64

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

def recognize_animal_in_image(image_path, command):
    """
    使用qwen-vl模型识别图片内容
    
    参数:
        image_path: 图片文件路径
        command: 识别指令/提示词
    
    返回:
        识别结果文本
    """
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return f"错误：图片文件 '{image_path}' 不存在！请检查文件路径。"
    
    # 使用Base64编码传输图片（避免Windows路径问题）
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # 根据文件扩展名确定MIME类型
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.png': 'png',
            '.gif': 'gif',
            '.bmp': 'bmp',
            '.webp': 'webp'
        }
        mime = mime_types.get(ext, 'jpeg')
        
        # 构建消息（使用Base64编码）
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'image': f'data:image/{mime};base64,{image_data}'
                    },
                    {
                        'text': command
                    }
                ]
            }
        ]
    except Exception as e:
        return f"读取图片文件失败: {str(e)}"
    
    try:
        response = MultiModalConversation.call(
            model='qwen-vl-plus', 
            messages=messages
        )
        
        # 输出响应信息
        if response.status_code == 200:
            result = response.output.choices[0].message.content
            return result
        else:
            return f"请求失败，错误码: {response.status_code}, 错误信息: {response.message}"
            
    except Exception as e:
        return f"发生错误: {str(e)}"


if __name__ == '__main__':
    # 图片路径
    image_path = 'animal.jpeg'
    
    print("=" * 60)
    print("使用阿里百炼qwen-vl模型识别图片中的动物")
    print("=" * 60)
    print(f"\n图片路径: {image_path}\n")
    
    # 调用识别函数
    result = recognize_animal_in_image(image_path, '请识别这张图片中的动物类型，直接输出动物类型比如狗，猫，不用分析')
    
    print("识别结果:")
    print("-" * 60)
    print(result)
    print("-" * 60)

