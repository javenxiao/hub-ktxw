from text_processor import load_and_split_doc
from vector_db import FaissVectorDB
from openai import OpenAI
from config import OPENAI_LLM_CONFIG, EMBEDDING_CONFIG

# 1. 文档路径不变
DOC_PATH = r"D:\438161609\WPS云盘\BADOU\week15\task2\downloads\东方证券-张庆-大语言模型在投研投顾中的应用与难点\full.md"

# 2. 初始化向量库
vector_db = FaissVectorDB()

# 3. 构建/加载向量库（逻辑不变）
text_chunks = load_and_split_doc(DOC_PATH)
vector_db.build_db(text_chunks)
# vector_db.load_db()  # 非首次运行用这个

# 4. 测试检索（逻辑不变）
query = "智能投顾的运营流程是怎样的？"
relevant_chunks = vector_db.search(query)
print("\n匹配到的相关文本块：")
for i, chunk in enumerate(relevant_chunks, 1):
    if i>3:
        break
    else:
        print(f"\n【chunk_num:{i}】{chunk[:100]}")

# 5. 替换为OpenAI大模型生成答案
def generate_answer(query: str, relevant_chunks: list[str]) -> str:
    client = OpenAI(
        api_key=EMBEDDING_CONFIG["openai_api_key"],
        base_url=EMBEDDING_CONFIG["base_url"]
    )
    prompt = f"基于以下参考内容，简洁准确回答问题：\n{''.join(relevant_chunks)}\n问题：{query}"
    response = client.chat.completions.create(
        model=OPENAI_LLM_CONFIG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=OPENAI_LLM_CONFIG["temperature"]
    )
    return response.choices[0].message.content

answer = generate_answer(query, relevant_chunks)
print(f"\n最终答案：{answer}")