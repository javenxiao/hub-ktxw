import requests
import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch,helpers
model = SentenceTransformer("./models/Qwen3-Embedding",trust_remote_code=True)
ELASTICSEARCH_URL = "http://localhost:9200"
es_client = Elasticsearch(ELASTICSEARCH_URL)
# 测试连接
if es_client.ping():
    print("连接成功！")
else:
    print("连接失败。请检查 Elasticsearch 服务是否运行。")

使用本地通过fastapi部署的mineru服务进行pdf解析
api_url = "http://localhost:8900/file_parse"
local_dir = "E:/PYH/output"
files = {'files': open('Dify大模型应用开发手册.pdf', 'rb'),'output':local_dir}
response = requests.post(api_url, files=files)
if response.status_code == 200:
    result = response.json()
    # 处理返回的解析结果
    print(result['results'])
else:
    print(f"解析失败: {response.status_code}")

# 初始化es索引
body={
        "mappings": {
            "properties": {
                "content": {"type": "text",
                            "analyzer":"ik_max_word",
                            "search_analyzer":"ik_smart"
                         },
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
index_name = "rag_dify"
if es_client.indices.exists(index=index_name):
    es_client.indices.delete(index=index_name)
    print(f"旧索引 '{index_name}' 已删除。")
es_client.indices.create(index=index_name,body=body)

#读取解析后的markdown文件
with open("E:/PYH/output/66e195cd-81f5-4a45-bff0-27c748a60345/Dify大模型应用开发手册/auto/Dify大模型应用开发手册.md","r",encoding="utf-8")as f:
    content = f.read()

#通过标题切分markdown文件
headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=True
        )
md_splits = markdown_splitter.split_text(content)

#切分chunk
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        )
for md_split in md_splits:
    chunks = text_splitter.split_text(md_split.page_content)
    for chunk in chunks:
        #清除原来markdown文件中的插入图片的格式
        pattern = r'!\[.*?\]\(.*?\)'
        chunk = re.sub(pattern, '', chunk)
        # 额外清理：可能存在的残留换行符和空格
        chunk = re.sub(r'\n\s*\n', '\n', chunk)
        chunk_vector = model.encode_document(chunk).tolist()    #用qwen-Embedding 编码
        es_client.index(index=index_name,document={"content":chunk,"content_vector":chunk_vector})  #存入ES
es_client.indices.refresh(index=index_name)  #刷新索引
