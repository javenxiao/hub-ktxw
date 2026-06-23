from elasticsearch import Elasticsearch
from agents import function_tool, Runner, Agent, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings,set_default_openai_api,set_tracing_disabled
import asyncio
import os
from sentence_transformers import SentenceTransformer
os.environ["OPENAI_API_KEY"] = "sk-90aa2b7df82745f3a46373cc0ddd0497"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
set_default_openai_api("chat_completions")
set_tracing_disabled(True)
@function_tool()
def search_es(query):
    model = SentenceTransformer("./models/Qwen3-Embedding",trust_remote_code=True)
    query_vector = model.encode_query(query).tolist()  #将查询编码为向量
    # 使用es进行向量检索
    es_client = Elasticsearch("http://localhost:9200")
    index_name = "rag_dify"
    body = {
        "knn":{
            "field":"content_vector",
            "query_vector":query_vector,
            "k":10,
            "num_candidates":30,
        },
        "fields":['content'],
        "_source":False
    }
    try:
        texts = []
        response = es_client.search(index=index_name,body=body)
        for hit in response['hits']['hits']:
            text = hit['fields']['content'][0]
            texts.append(text)
        result = "\n\n".join(texts)
        return result
    except Exception as e:
        return ""
async def main(question:str):
    external_client = AsyncOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_BASE_URL']
    )
    search_agent = Agent(
        name="Search Agent",
        instructions="调用工具对用户的问题在es中进行检索相关文档，总结并返回和原本问题相关的结果",
        model=OpenAIChatCompletionsModel(model="qwen-max",
                                         openai_client=external_client),
        tools=[search_es],
        model_settings=ModelSettings(parallel_tool_calls=False)
    )
    summary_agent = Agent(
        name="Summary Agent",
        instructions="根据用户的提问,总结从es中返回的结果，并将其整合为一个合理的回答，如果根据返回结果无法回答用户问题，则回答根据已知内容无法回答，不要有其他多余的输出",
        model=OpenAIChatCompletionsModel(model="qwen-max",
                                        openai_client=external_client)
    )
    search_result = await Runner.run(search_agent,question)
    search_result = search_result.final_output
    result = await Runner.run(summary_agent,f"这是用户的问题{question}\n这是es返回的搜索结果{search_result}")
    print(result.final_output)
if __name__ =="__main__":
    asyncio.run(main("Dify如何使用"))
