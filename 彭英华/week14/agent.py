import asyncio
from sentence_transformers import SentenceTransformer
import json
from agents.mcp.util import ToolFilterStatic
import torch
from agents import Agent,AsyncOpenAI,OpenAIChatCompletionsModel,ModelSettings,Runner
from agents.mcp.server import MCPServerSse
from agents.mcp import MCPServer
import os
from openai.types.responses import ResponseTextDeltaEvent
os.environ["OPENAI_API_KEY"] = "sk-90aa2b7df82745f3a46373cc0ddd0497"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
with open ("formulas_dict.json","r",encoding="utf-8") as f:
    data = json.load(f)
# 简易RAG，通过Qwen-Embedding进行编码，然后通过相似度计算top3
corpus = [formula['formula_description'] for formula in data]
embedder = SentenceTransformer("./models/Qwen3-Embedding",trust_remote_code=True)
query = input("请输入你想查询的：")
available_tools=[]
query_embedding = embedder.encode_query(query,convert_to_tensor=True)
with open ("formulas_dict.json","r",encoding="utf-8") as f:
    data = json.load(f)
corpus_embeddings = embedder.encode_document(corpus,convert_to_tensor=True)
similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
scores, indices = torch.topk(similarity_scores, k=3)
for idx in indices:
    available_tools.append(data[idx]['tool_name'])
use_tools = ToolFilterStatic(allowed_tool_names=available_tools)
print(use_tools)
# Agent
async def run(mcp_server:MCPServer):
    exteral_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                 base_url=os.environ["OPENAI_BASE_URL"])
    agent = Agent(name="Assistant",
                  instructions="你是一个擅长调用各种计算公式的专家，如果用户给的参数不够公式计算，则回答无法回答",
                  mcp_servers=[mcp_server],
                  model=OpenAIChatCompletionsModel(model="qwen-max",
                                                   openai_client=exteral_client),
                  model_settings=ModelSettings(parallel_tool_calls=False)
                  )
    result =  Runner.run_streamed(agent,query)
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):  # 如果是大模型的回答
                if event.data.delta:
                    print(event.data.delta, end="", flush=True)
async def main():
    async with MCPServerSse(name="SSE Python Server",
                            params={
                                "url":"http://localhost:8900/sse",
                                    },
                            tool_filter=use_tools,
                            client_session_timeout_seconds=20
                            ) as server:
        await run(server)
if __name__ == "__main__":
    asyncio.run(main())
