from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from openai import OpenAI
import os

model = SentenceTransformer("../models/BAAI/bge-small-zh-v1.5/")

client = MilvusClient(
    uri="https://in03-b778dd7b2a6eb86.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
    token="97913c8d6bc421eddd5469c4a8a545fb72b900cd502eb68842f987ae455d1cda60a38815e4d2635981e1d115768fd2956bd5ec78"
)

openai_client = OpenAI(
    api_key="sk-88d3ca9887854f7e92a8485baa49993f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def insert_data(data: str, page: int, file: str):
    features = get_text_features(data)
    data = [
        {
            "vector": features,
            "content": data,
            "page": page,
            "file": file
        }
    ]
    insert_result = client.insert(
        collection_name="pdf_vector",
        data=data,
        auto_id=True
    )
    return insert_result['ids'][0]


def read_md_files(path: str):
    # 读取路径下的md文件内容
    for file in os.listdir(path):
        if file.endswith('.md'):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                content = f.read()
            insert_data(content, 0, os.path.join(path, file))


def retrieve_file(query):
    results = client.search(
        collection_name="pdf_vector",  # 对哪一个collection搜索
        anns_field="vector",  # 对哪一个一段进行排序
        limit=3,  # 返回多少个
        # 打分方法
        search_params={
            "metric_type": "COSINE"
        },
        output_fields=["file"],
        data=[get_text_features(query)]
    )
    # 提取其中的file
    files = [result['entity']['file'] for result in results[0]]
    # 读取md文件内容
    file_content_dict = {}
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            file_content_dict[file] = f.read()

    return file_content_dict


def get_text_features(text):
    return model.encode(text, normalize_embeddings=True)

def qa(query):
    retrieved_content = retrieve_file(query)
    model_results = openai_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": f"""
            帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。
资料：

{retrieved_content}

问题：{query}
            """
             }
        ]
    )
    return model_results.choices[0].message.content

if __name__ == "__main__":
    qa("尼康z6怎么打开hdr模式")

