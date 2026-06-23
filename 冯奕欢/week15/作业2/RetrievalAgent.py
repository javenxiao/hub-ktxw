from openai import OpenAI
from EmbeddingManager import ChromaEmbeddingManager


class RetrievalAgent:

    def __init__(self):
        self.client = OpenAI(
            api_key="xxxxxxxxxxxxxxxxxxxx",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.em = ChromaEmbeddingManager()

    def query_qa(self, question):
        """
        回答问题
        :param question 用户提问
        :return:
        """
        retrieve_result = self.em.rag_retrieve(question, top_k=3)
        print(retrieve_result)
        retrieve_text = "\n".join([" - " + result["content"] for result in retrieve_result])
        completion = self.client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "system", "content": f"""
你是一个文档问答助手，根据文档检索的结果，回答用户问题。如果检索的结果与问题不相关，回答暂无结果。
文档检索结果：
{retrieve_text}
                """},
                {"role": "user", "content": f"""
用户提问：{question}
                """}
            ]
        )
        data = completion.choices[0].message.content
        print(data)
        return data
