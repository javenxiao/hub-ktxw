from elasticsearch import Elasticsearch
import json

es = Elasticsearch("http://192.168.124.36:9200")
if es.ping():
    print("Elasticsearch is up")
else:
    print("Elasticsearch is not up")

index_name = 'user_index'

if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "username": {"type": "text"},
                    "password": {"type": "text"},
                    "phone_number": {"type": "text"}
                }
            }
        })
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已存在。")

doc_1 = {
    "username": "admin",
    "password": "123456",
    "phone_number": "13800000000"
}

es.index(index=index_name, document=doc_1)
print(f"文档 {doc_1['username']} 已成功索引。")

es.indices.refresh(index=index_name)
res_1 = es.search(
    index=index_name,
    body={
        "query": {
            "match_all": {}
        }
    }
)

print(f"找到 {res_1['hits']['total']['value']} 条文档。")
for hit in res_1['hits']['hits']:
    print(f"文档内容：{json.dumps(hit['_source'], ensure_ascii=False, indent=2)}")
