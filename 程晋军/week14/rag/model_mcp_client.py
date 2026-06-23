import asyncio
import json
import os
import re
import subprocess
import time
import signal
import psutil
from openai import OpenAI
from fastmcp import Client


# 之后直接绝对导入
from config import QWEN_EMBEDDING_CONFIG  # 正确！

# 默认配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "gen_mcp_tool", "mcp_tool_config.json")


def load_tool_config(config_path=CONFIG_FILE):
    """加载MCP工具配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def kill_existing_mcp_servers():
    """杀死所有正在运行的MCP服务器进程"""
    killed_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 检查进程是否是Python进程并且命令行中包含MCP相关关键词
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                cmdline = proc.info['cmdline']
                # 检查命令行参数
                cmd_line_str = ' '.join(cmdline).lower()
                if ('mcp_server' in cmd_line_str or
                        'calculate_' in cmd_line_str or
                        'fastmcp' in cmd_line_str):
                    # 先尝试优雅地终止进程
                    proc.send_signal(signal.SIGTERM)
                    time.sleep(1)
                    # 如果进程仍存在，则强制终止
                    if proc.is_running():
                        proc.kill()
                    killed_processes.append(proc.info['pid'])
                    print(f"已终止进程 PID: {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return killed_processes


def extract_function_info(server_file_path, function_name):
    """从服务器文件中提取函数信息（参数和文档说明）"""
    if not os.path.exists(server_file_path):
        raise FileNotFoundError(f"服务器文件未找到: {server_file_path}")

    with open(server_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取函数定义和文档字符串
    func_pattern = rf'def\s+{function_name}\s*\(([^)]+)\)(.*?)(?:\n\s*@|\n\w|\Z)'
    func_match = re.search(func_pattern, content, re.DOTALL)

    if not func_match:
        raise ValueError(f"未找到函数 {function_name}")

    # 解析参数
    params_str = func_match.group(1)
    params = []

    # 简单参数解析
    for param in params_str.split(','):
        param = param.strip()
        if ':' in param:
            param_name = param.split(':')[0].strip()
        else:
            param_name = param
        params.append(param_name)

    # 提取文档字符串
    docstring_match = re.search(r'""".*?"""', func_match.group(2), re.DOTALL)
    docstring = docstring_match.group(0) if docstring_match else ""

    return {
        'params': params,
        'docstring': docstring
    }


def generate_sample_params_with_qwen(params, docstring):
    """使用Qwen大模型根据函数说明生成参数的默认数值"""
    # 初始化Qwen客户端
    client = OpenAI(
        api_key=QWEN_EMBEDDING_CONFIG["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=30
    )

    # 构造提示词
    prompt = f"""
请根据以下函数文档说明，为每个参数生成合理的默认数值。

要求：
1. 为每个参数生成一个浮点数数值
2. 数值应该符合参数的实际含义
3. 以JSON格式返回结果，格式为{{"param_name": value, ...}}
4. 不要包含任何其他文字，只返回JSON

函数文档说明：
{docstring}

参数列表：
{', '.join(params)}

请生成合理的默认数值：
"""

    try:
        # 调用Qwen大模型生成参数值
        completion = client.chat.completions.create(
            model="qwen-flash",
            messages=[
                {"role": "system", "content": "你是一个专业的Python程序员，熟悉数学模型和参数设置。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        if completion and completion.choices and completion.choices[0].message:
            response_text = completion.choices[0].message.content.strip()
            # 尝试解析JSON
            try:
                # 提取可能的JSON部分
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    return json.loads(json_text)
                else:
                    # 如果没有找到JSON，使用默认值
                    return {param: 1.0 for param in params}
            except json.JSONDecodeError:
                # 如果解析失败，使用默认值
                return {param: 1.0 for param in params}
        else:
            # 如果模型没有返回内容，使用默认值
            return {param: 1.0 for param in params}

    except Exception as e:
        print(f"调用Qwen模型生成参数时出错: {e}")
        # 出错时使用默认值
        return {param: 1.0 for param in params}


def start_mcp_server(server_file):
    """启动MCP服务器"""
    # 杀死所有现有的MCP服务器进程
    killed_processes = kill_existing_mcp_servers()
    if killed_processes:
        print(f"已终止 {len(killed_processes)} 个现有的MCP服务器进程")
        # 等待一段时间确保端口释放
        time.sleep(2)

    try:
        # 启动服务器进程
        server_process = subprocess.Popen(
            ["python", server_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # 等待服务器启动
        time.sleep(3)

        # 检查进程是否还在运行
        if server_process.poll() is None:
            print(f"MCP服务器已启动: {server_file}")
            return server_process
        else:
            # 进程已结束，可能是启动失败
            stdout, stderr = server_process.communicate()
            print(f"服务器启动失败: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"启动服务器时出错: {e}")
        return None


# 读取配置文件
try:
    config = load_tool_config()
    # 根据配置创建客户端实例
    client = Client("http://localhost:8002/mcp")
    function_name = config["function_name"]
    server_file = os.path.join("gen_mcp_tool", config["mcp_tool_filename"])
except Exception as e:
    print(f"加载配置文件失败: {e}")
    config = None
    function_name = None
    server_file = None


def get_user_params(auto_params):
    """获取用户输入的参数，如果用户未提供则使用自动生成的参数"""
    user_params = {}
    print("\n是否要手动输入参数值？(y/n，默认为n，使用自动生成的参数): ")
    use_manual = input().strip().lower()
    
    if use_manual == 'y' or use_manual == 'yes':
        print("请输入参数值（直接回车使用自动生成的默认值）:")
        for param_name, default_value in auto_params.items():
            print(f"  {param_name} (默认: {default_value}): ", end="")
            user_input = input().strip()
            if user_input == "":
                user_params[param_name] = default_value
            else:
                try:
                    # 尝试转换为浮点数
                    user_params[param_name] = float(user_input)
                except ValueError:
                    print(f"    输入无效，使用默认值: {default_value}")
                    user_params[param_name] = default_value
    else:
        user_params = auto_params
        print("使用自动生成的参数值")
    
    return user_params

async def demo_generated_model():
    """演示调用动态生成的模型函数"""
    # 检查配置是否成功加载
    if config is None or function_name is None or server_file is None:
        print("配置未正确加载，无法启动MCP服务器")
        return
        
    # 启动MCP服务器
    server_process = start_mcp_server(server_file)
    if not server_process:
        print("无法启动MCP服务器，使用默认参数进行演示")
        return

    try:
        # 等待服务器完全启动
        time.sleep(2)

        # 提取函数参数和文档说明
        func_info = extract_function_info(server_file, function_name)
        param_names = func_info['params']
        docstring = func_info['docstring']

        # 使用Qwen大模型生成示例参数
        auto_params = generate_sample_params_with_qwen(param_names, docstring)
        
        # 获取用户输入的参数或使用自动生成的参数
        params = get_user_params(auto_params)

        async with client:
            print(f"1. 调用函数: {function_name}")
            print(f"   函数说明: {docstring[:100]}...")
            print(f"   使用参数: {params}")
            result = await client.call_tool(function_name, params)
            # 只显示结果中的数值部分
            if hasattr(result, 'structured_content') and 'result' in result.structured_content:
                print(f"   结果: {result.structured_content['result']}")
            elif hasattr(result, 'data'):
                print(f"   结果: {result.data}")
            else:
                print(f"   结果: {result}")
            print()
    except Exception as e:
        print(f"调用函数时出错: {e}")


if __name__ == "__main__":
    # 优先运行动态生成的模型演示
    asyncio.run(demo_generated_model())

    # 运行所有模型演示
    # asyncio.run(demo_all_models())