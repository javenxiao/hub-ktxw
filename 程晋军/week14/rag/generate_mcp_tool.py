import os
import argparse
import re
import json
from openai import OpenAI
from config import QWEN_EMBEDDING_CONFIG

def generate_mcp_tool_from_md(md_file_path, output_folder_path):
    """根据MD文件生成MCP工具代码"""
    # 读取MD文件内容
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 初始化Qwen客户端
    client = OpenAI(
        api_key=QWEN_EMBEDDING_CONFIG["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=30
    )
    
    # 构造提示词
    prompt = f"""
请根据以下Markdown格式的数学模型文档，生成一个符合FastMCP规范的工具函数。

要求：
1. 使用@mcp.tool装饰器
2. 函数名以calculate_开头，后接有意义的名称
3. 函数文档字符串需要包含详细的中文说明，解释模型背景、公式含义和参数说明
4. 参数名使用有意义的英文单词
5. 返回计算结果（浮点数）
6. 函数体实现给定的数学公式
7. 只返回函数代码，不要有任何额外的说明文字
8. 不要使用代码块标记（如```python）

参考格式：
@mcp.tool
def calculate_function_name(param1: float, param2: float) -> float:
    \"""
    函数说明
    
    公式: 具体的数学公式
    
    Args:
        param1: 参数说明
        param2: 参数说明
    
    Returns:
        float: 返回值说明
    \"""
    # 实现公式计算
    return result

Markdown文档内容：
{md_content}
"""
    
    try:
        # 调用Qwen大模型生成代码
        completion = client.chat.completions.create(
            model="qwen-flash",
            messages=[
                {"role": "system", "content": "你是一个专业的Python程序员，熟悉MCP (Model Control Protocol) 工具开发。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        if completion and completion.choices and completion.choices[0].message:
            tool_code = completion.choices[0].message.content.strip()
        else:
            raise Exception("模型返回空内容")
            
    except Exception as e:
        print(f"调用Qwen模型生成代码时出错: {e}")
        return False
    
    # 从生成的代码中提取函数名作为文件名
    function_name_match = re.search(r'def\s+(calculate_\w+)\s*\(', tool_code)
    if function_name_match:
        filename = function_name_match.group(1)
    else:
        # 默认文件名
        filename = "generated_mcp_tool"
    
    # 生成完整的MCP服务器代码
    server_code = f'''from fastmcp import FastMCP
import math

# 创建 MCP 服务器实例
mcp = FastMCP("Formula Calculation Server")

{tool_code}

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run(transport="http", port=8002)
'''
    
    # 确保输出文件夹存在
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 构造完整的输出文件路径
    output_file_path = os.path.join(output_folder_path, f"mcp_server_{filename}.py")
    
    # 写入文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(server_code)
    
    # 将文件名保存到JSON文件中
    json_output_path = os.path.join(output_folder_path, "mcp_tool_config.json")
    config_data = {
        "mcp_tool_filename": f"mcp_server_{filename}.py",
        "function_name": filename
    }
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    print(f"MCP服务器代码已生成: {output_file_path}")
    print(f"配置信息已保存到: {json_output_path}")
    return True