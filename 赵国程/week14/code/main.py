import glob
import json
import asyncio
import torch
import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.mcp import MCPServerSse, ToolFilterStatic
from sentence_transformers import SentenceTransformer, util
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from tqdm import tqdm

async def main():
    doc_to_function_map = {
        "0e78a7e1-0607-43a8-abd1-ff8c77e0bfa.pdf": "simulate_linear_dynamic_system",
        "0daef473-e660-4984-be4d-940433aa889.pdf": "predict_cattle_population",
        "0d9d4e8c-51c1-474b-b8e2-8b950e53437.pdf": "calculate_influence",
        "0d2f19ba-1875-4057-b804-379367fedec.pdf": "evaluate_quadratic_model",
        "0bcccdd0-b9a4-4f9b-afc2-d14b4384098.pdf": "predict_system_state",
        "0ba15b17-85d2-4944-9a04-a9bd23c2e3f.pdf": "predict_learning_score",
        "0b579473-43f3-4f7d-a45a-312089b766a.pdf": "moisture_content",
        "0b9a5088-426e-4c65-8bfb-ef98fc410da.pdf": "predict_crop_yield",
        "0a96883f-a7ee-4aa7-b020-efdb1f634d0.pdf": "predict_daily_orders",
        "0a948fc4-b083-44c6-af02-70be51108f7.pdf": "simulate_dissolved_oxygen"
    }

    # 提前固定顺序
    filenames = list(doc_to_function_map.keys())
    contents = []
    document_path = "/Users/zhaoguocheng/PycharmProjects/nlp_course/task14/documents"

    for filename in tqdm(filenames):
        file_path = os.path.join(document_path, filename)
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            full_text = "\n".join(doc.page_content for doc in documents)
            contents.append(full_text)
        else:
            contents.append("")

    print(len(contents))

    model_name = '/Users/zhaoguocheng/PycharmProjects/nlp_course/models/BAAI/bge-small-zh-v1.5'
    embedder = SentenceTransformer(model_name, trust_remote_code=True)

    external_client = AsyncOpenAI(
        api_key="sk-88d3ca9887854f7e92a8485baa49993f",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    corpus_embeddings = embedder.encode(contents, convert_to_tensor=True)

    # 查询 & 相似度
    query = "当x=2时，二次函数y = 2*x^2 + 3*x + 1的值是多少？"
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    similar_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    top_k = min(8, len(corpus_embeddings))
    scores, indices = torch.topk(similar_scores, k=top_k)

    # 正确获取工具名
    tools = []
    for idx in indices:
        idx = idx.item()
        tools.append(doc_to_function_map[filenames[idx]])

    tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools)
    mcp_server = MCPServerSse(
        name="qa_server",
        params={"url": "http://localhost:8090/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )
    async with mcp_server:
        system_prompt = """
                你是一个专业的数学助手，你可以使用以下工具来回答用户的问题：
            """
        agent = Agent(
            name="math_agent",
            instructions=system_prompt,
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model="qwen-max",
                openai_client=external_client
            ),
            model_settings=ModelSettings(
                parallel_tool_calls=False
            )
        )
        result = Runner.run_streamed(agent, input=query)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
