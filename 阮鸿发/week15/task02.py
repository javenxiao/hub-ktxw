import os
os.environ["OPENAI_API_KEY"] = "sk-247b4c94005a487b9516879348c4bcec"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import numpy as np

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

# 读取文档内容
with open('./模型论文/2509-MinerU_01/auto/2509-MinerU_01.md', 'r', encoding='utf-8') as f:
    document_content = f.read()

# 将文档分割成段落或句子作为检羃单元
paragraphs = [para.strip() for para in document_content.split('\n\n') if para.strip()]

async def rag_call():
    tfidf = TfidfVectorizer()
    # 对文档段落进行向量化
    paragraphs_tfidf = tfidf.fit_transform(
        [" ".join(jieba.lcut(para)) for para in paragraphs if para.strip()]
    ).toarray()

    user_question = "Document parsing [57] serves as a fundamental task in multimodal understanding"
    user_question_tfidf = tfidf.transform(
        [" ".join(jieba.lcut(user_question))]
    ).toarray()

    # 计算相似度并获取最相关的段落
    similarities = np.dot(user_question_tfidf, paragraphs_tfidf.T)[0]
    top_indices = np.argsort(similarities)[::-1][:5]

    print("Top 5 relevant paragraphs:")
    for i in top_indices:
        print(f"Paragraph {i}: {paragraphs[i][:100]}...")
    
    # 构造上下文
    context = "\n\n".join([paragraphs[i] for i in top_indices])
    
    # 使用大模型进行问答
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    prompt = f"""
    请根据以下文档内容回答问题：
    
    文档内容：
    {context}
    
    问题：{user_question}
    
    请首先复述用户问题，然后根据文档内容直接输出结果，最后进行总结。
    """
    
    response = await client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    # 流式输出回答
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(rag_call())
