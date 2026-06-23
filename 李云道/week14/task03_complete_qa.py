import asyncio
from glob import glob

import pdfplumber
import torch
from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, SQLiteSession, Runner
from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import Client, AsyncClient
from openai.types.responses import ResponseTextDeltaEvent
from sentence_transformers import SentenceTransformer

'''
结合rag tool筛选 加 mcp tool执行完成 整个问答。
  - 步骤2: 定义mcp服务，并且将每一个公式整理为mcp 的一个可执行的tool 或 sympy的计算过程。
  - 步骤3: 得到用户提问的时候，需要选择对应的tool
    - 通过rag的步骤（用户的提问 与 公式 进行相似度计算，也可以加入rerank 过程，筛选top1/3公式） -》 tool 白名单
  - 步骤4: 调用对应的tool，执行，得到结果，汇总得到回答。
'''

# 设置API Key
api_key = "sk-2ff484a65dbd47668f71c459353fd8ff"
api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


async def qa(query: str, allowed_tool_names: list):
    external_client = AsyncClient(
        api_key=api_key,
        base_url=api_url,
    )
    session = SQLiteSession(__name__)
    tool_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=allowed_tool_names)
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        client_session_timeout_seconds=20,
        tool_filter=tool_filter)

    async with mcp_server:
        agent = Agent(
            name="assistant",
            instructions="你是一个专业的数学计算助手，负责提取用户的问题中的参数，并调用相应的数学公式进行计算，最后给出详细的解答过程和结果。",
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model="qwen-max",
                openai_client=external_client,
            ),
            tool_use_behavior="run_llm_again",
            model_settings=ModelSettings(parallel_tool_calls=False)
        )
        result = Runner.run_streamed(agent, input=query, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)


def load_pdf(pdf_path):
    try:
        formulas = []
        for file_path in glob(pdf_path):
            pdf = pdfplumber.open(file_path)
            content = ""
            for page in pdf.pages:
                text = page.extract_text()
                content += text.strip()
            formulas.append(content)
        return formulas, len(formulas)
    except FileNotFoundError:
        raise FileNotFoundError(f"目录不存在: {pdf_path}")


if __name__ == '__main__':
    # 用户提问与提示词
    query = "学生张三的情况如下：学习时常25小时，出勤率95%，平时测验平均分65%，课堂参与度4分。他的综合表现分数是多少"

    # 读取公式库
    formulas, formulas_size = load_pdf("./pdf/*.pdf")
    formulas_to_mcp_dict = {
        0: 'get_dissolved_oxygen_concentration',
        1: 'get_daily_orders_prediction',
        2: 'get_crop_yield',
        3: 'get_cumulative_evaporation',
        4: 'get_student_performance_score',
        5: 'get_time_series_prediction',
        6: 'get_quadratic_deterministic_output',
        7: 'get_cultural_influence_score',
        8: 'get_cattle_population_growth',
        9: 'get_multi_input_dynamic_system'
    }

    # RAG 公式检索，选择合适的公式
    model_name = r"D:\Download\BAAI\bge-small-zh-v1.5"
    model = SentenceTransformer(model_name)
    formulas_embeddings = model.encode(formulas, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity_scores = model.similarity(query_embedding, formulas_embeddings)
    index = torch.topk(similarity_scores, k=(formulas_size // 3)).indices.tolist()[0]

    # 选择对应的mcp tool
    asyncio.run(qa(query=query, allowed_tool_names=[formulas_to_mcp_dict[i] for i in index]))
