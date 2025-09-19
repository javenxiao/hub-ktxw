from datetime import datetime

from elasticsearch import Elasticsearch
elastic_url = "http://localhost:9200"
elastic = Elasticsearch(elastic_url)
if elastic.ping():
    print("连接成功！")
else:
    print("连接失败。请检查 Elasticsearch 服务是否运行。")
index_name = "mydatabase"
documents = [
    {
      "title": "Elasticsearch 入门指南",
      "content": "这是一篇关于如何安装和使用 Elasticsearch 的详细文章。学习搜索技术可以提升数据处理能力。",
      "tags": ["Elasticsearch", "教程", "搜索"],
      "author": "张三",
      "created_at": datetime(2023, 10, 26, 10, 0, 0)
    },
    {
      "title": "深入理解IK分词器",
      "content": "IK分词器是中文分词的优秀工具。它的智能分词和最细粒度分词模式各有优势。",
      "tags": ["分词", "IK", "中文"],
      "author": "李四",
      "created_at": datetime(2023, 10, 25, 15, 30, 0)
    }
]
body = {
    'settings':{
        'number_of_shards':3,
        'number_of_replicas':0
    },
    'mappings':{
        'properties':{
            'title':{
                'type':'text',
                'analyzer':'ik_max_word',
                'search_analyzer':'ik_smart'
            },
            'content':{
                'type':'text',
                'analyzer':'ik_max_word',
                'search_analyzer':'ik_smart'
            },
            'tags':{
                'type':'keyword'
            },
            'author':{
                'type':'keyword'
            },
            'created_at':{
                'type':'date'
            }
        }
    }
}
if not elastic.indices.exists(index = index_name):
    elastic.indices.create(index = index_name,body = body)
else:
    print("索引已存在")
for doc in documents:
    elastic.index(index=index_name, document=doc)
    print(f"文档已插入: '{doc['title']}'")
elastic.indices.refresh(index = index_name)
def search_docs(query):
    response = elastic.search(index=index_name, body=query)
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    for hit in response['hits']['hits']:
        print(f"得分：{hit['_score']}，文档：{hit['_source']['title']}")
query = {
    'query':{
        'bool':{
            'must':{
                'match':{
                    'title':'分词',
                }
            },
            'filter':{
                'term':{'content':'中文'}
            }
       }
    }
}
# search_docs(query)
print(elastic.count(index=index_name))
body_1 = {
    "aggs": {
        "unique_titles": {
            "cardinality": {
                "field": "author"
            }
        }
    },
    'size':0
}
result = elastic.search(index=index_name, body=body_1)
print(result['aggregations']['unique_titles'])
