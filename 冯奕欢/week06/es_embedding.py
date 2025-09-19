import time

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

INDEX_NAME = "embedding"

INDEX_MAPPING = {
    'mappings': {
        'properties': {
            'content': {
                'type': 'text'
            },
            'embedding': {
                'type': 'dense_vector',
                'dims': 512,
                'index': True,
                'similarity': 'cosine'
            }
        }
    }
}

# 地址
hosts = [{
    'host': 'localhost',
    'port': 9200,
    'scheme': 'http'
}]

# 认证信息
auth = ('elastic', 'qhNdREadcp6ECGacW6Iw')

# 创建ES
es = Elasticsearch(hosts=hosts, basic_auth=auth)

# 测试连接
if not es.ping():
    print('Elasticsearch连接失败')

print('Elasticsearch连接成功')

# 创建索引
if not es.indices.exists(index=INDEX_NAME):
    try:
        es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
        print('Elasticsearch创建索引成功 -> ', INDEX_NAME)
    except Exception as e:
        print('Elasticsearch创建索引失败 -> ', INDEX_NAME, " - ", e)
else:
    print('Elasticsearch索引已存在 -> ', INDEX_NAME)

# 内容
texts = [
    "人工智能是未来的趋势。",
    "机器学习是人工智能的一个重要分支。",
    "自然语言处理技术让机器理解人类语言。",
    "今天天气真好，适合出去玩。",
    "我最喜欢的运动是篮球和足球。"
]

# 加载模型
model = SentenceTransformer('../models/BAAI/bge-small-zh-v1.5')

# 插入文档
documents = [
    {
        'text': text,
        'embedding': model.encode(text).tolist()
    }
    for text in texts
]

# 插入文档
for document in documents:
    try:
        es.index(index=INDEX_NAME, body=document)
        print('插入文档成功 -> ', INDEX_NAME, " - ", document)
    except:
        print('插入文档失败 -> ', INDEX_NAME, " - ", document)

# 刷新索引
es.indices.refresh(index=INDEX_NAME)
# 等待索引刷新
time.sleep(1)

# 搜索
query_text = "关于AI和未来的技术"
query_embedding = model.encode(query_text)
query = {
    'knn': {
        'field': 'embedding',
        'query_vector': query_embedding,
        'k': 3,
        'num_candidates': 10
    },
    'fields': ['text'],
    '_source': False
}
response = es.search(index=INDEX_NAME, body=query)

print(f"查询文本: '{query_text}'")
print(f"找到 {response['hits']['total']['value']} 个最相关的结果:")

for hit in response['hits']['hits']:
    score = hit['_score']
    text = hit['fields']['text'][0]
    print(f"得分: {score:.4f}, 内容: {text}")


response_combined = es.search(
    index=INDEX_NAME,
    body={
        "knn": {
            "field": "text_vector",
            "query_vector": query_embedding,
            "k": 3,
            "num_candidates": 10
        },
        "query": {
            "match": {
                "text": "技术"
            }
        },
        "fields": ["text"],
        "_source": False
    }
)

print(f"查询文本: '{query_text}' (并过滤包含 '技术' 的文档)")
print(f"找到 {response_combined['hits']['total']['value']} 个最相关的结果:")

for hit in response_combined['hits']['hits']:
    score = hit['_score']
    text = hit['fields']['text'][0]
    print(f"得分: {score:.4f}, 内容: {text}")


