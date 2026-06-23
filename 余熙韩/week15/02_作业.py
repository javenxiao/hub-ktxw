import os
import requests
import asyncio
import logging
import subprocess
import tempfile
import json
from pathlib import Path
from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, ItemHelpers
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 配置 MinerU API 密钥和接口地址
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIyMDUwMDk3OSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2NTQzODIxNCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMGU4OWY3YWItNDQyNC00YjEwLWExNjktOTExYWRlNzg5OTdmIiwiZW1haWwiOiIiLCJleHAiOjE3NjY2NDc4MTR9.cwcRdL8xvAf3rjK9VJhP6hyA1ZpedCU_w45zj4q1ULMn3V_73aFhNC0Gx2tZN4Dc_zXpy9keyBQ2WyAI4i3dVg")
MINERU_API_URL = "https://mineru.net/api/v4/extract/task"

os.environ["OPENAI_API_KEY"] = "sk-0fce7eb19bc042469a27ea5005b65635"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"



def parse_document(file_path: str) -> str:
    """
    调用本地 MinerU 解析 PDF/Word 文档，返回纯文本内容
    """
    logger.info(f"开始解析文档: {file_path}")

    if not os.path.isfile(file_path):
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 创建临时目录存放解析结果
    output_dir = "./output"
    # 调用本地 mineru 命令行工具
    cmd = ["mineru", "-p", file_path, "-o", str(output_dir)]
    logger.info(f"执行命令: {' '.join(cmd)}")

    try:
        # 添加超时控制
        result = subprocess.run(cmd,
                                check=True,
                                capture_output=True,
                                text=True,
                                timeout=300)  # 5分钟超时
        logger.info(f"命令执行成功，返回码: {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.error(f"MinerU 解析超时 (5分钟)")
        raise RuntimeError("本地 MinerU 解析超时")
    except subprocess.CalledProcessError as e:
        logger.error(f"本地 MinerU 解析失败，返回码: {e.returncode}, 错误输出: {e.stderr}")
        raise RuntimeError(f"本地 MinerU 解析失败: {e.stderr}")
    except Exception as e:
        logger.error(f"执行 MinerU 命令时发生意外错误: {str(e)}")
        raise
    output_dir = f"{output_dir}/{Path(file_path).stem}/auto"
    print(output_dir)
    # 读取生成的 markdown 文件内容

    md_files = list(Path(output_dir).glob("*.md"))
    if not md_files:
        logger.error(f"未在输出目录找到 markdown 文件: {output_dir}")
        raise RuntimeError("本地 MinerU 未生成 markdown 文件")

    logger.info(f"找到 {len(md_files)} 个 markdown 文件")
    text_content = ""
    for md_file in md_files:
        logger.info(f"读取文件: {md_file}")
        text_content += md_file.read_text(encoding="utf-8") + "\n"

    logger.info(f"文档解析完成，共获取字符数: {len(text_content)}")
    return text_content.strip()



async def rag_qa(text: str, question: str):
    """
    将解析后的文档文本作为上下文，通过 OpenAI 代理调用 Qwen 大模型进行 RAG 问答
    使用 SSE（Server-Sent Events）流式连接获取回答
    """
    query = ("你是智能问答助手。以下背景信息来自用户上传的文档，请基于该信息回答问题，"
                     "若信息不足请明确说明，禁止编造答案。\n\n"
                     f"问题：{question}\n\n"
                     f"背景信息：\n{text}\n\n")
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    rag_agent = Agent(name="RAG agent",
                      instructions="你是一个智能问答助手，基于提供的背景信息回答问题。",
                      model=OpenAIChatCompletionsModel(
                          model="qwen-max", openai_client=external_client),
                      model_settings=ModelSettings(parallel_tool_calls=False)
                      )
    result = await Runner.run(rag_agent, query)
    return result.final_output


async def main(question: str):
    """
    主函数：解析文档并执行 RAG 问答
    """
    logger.info("程序启动")
    logger.info(f"用户问题: {question}")

    # 示例：解析本地文档并执行 RAG 问答
    doc_path = "0a948fc4-b083-44c6-af02-70be51108f7.pdf"  # 可替换为 .docx 等
    logger.info(f"文档路径: {doc_path}")

    try:
        doc_text = parse_document(doc_path)
        # logger.info("文档解析完成")
        # 测试数据：MinerU2.5 论文摘要（示例）
        # doc_text = (
        #     "MinerU2.5 是一个基于深度学习的文档解析框架，支持 PDF、Word 等多种格式。"
        #     "它通过视觉-语言模型提取文本、表格与公式，并输出结构化 Markdown。"
        #     "相比上一代，MinerU2.5 在中文版面分析准确率提升 12%，公式识别 F1 达到 0.93。"
        #     "系统支持本地部署，提供 RESTful API，方便二次开发。"
        # )
        logger.info("开始 RAG 问答")
        answer = await rag_qa(doc_text, question)

        print("\n问答结果：")
        print(answer)
        logger.info("程序执行完成")
    except Exception as e:
        logger.error(f"运行出错：{str(e)}", exc_info=True)
        print("运行出错：", e)


if __name__ == "__main__":
    # question = input("请输入您的问题：")
    question = "解析里面的公式"
    asyncio.run(main(question))
