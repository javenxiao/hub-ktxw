from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json

# 创建Elasticsearch客户端
es = Elasticsearch(["http://localhost:9200"])

# 检查连接
if es.ping():
    print("成功连接到Elasticsearch")
else:
    print("无法连接到Elasticsearch，请确保Elasticsearch正在运行")
    # 退出或跳过ES相关代码
    exit(1)

    # 定义Elasticsearch索引映射
    index_mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
                "abstract": {"type": "text", "analyzer": "standard"},
                "authors": {"type": "text", "analyzer": "standard"},
                "publish_date": {"type": "date"},
                "category": {"type": "keyword"},
                "citations": {"type": "integer"},
                "keywords": {"type": "text", "analyzer": "standard"},
                "created_at": {"type": "date"}
            }
        }
    }

    # 创建索引（如果不存在）
    index_name = "research_papers"
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_mapping)
        print(f"创建索引: {index_name}")


    # 从SQLite获取数据并导入到Elasticsearch
    def index_papers_to_es():
        # 连接SQLite数据库
        conn = sqlite3.connect('experiment_data.db')
        cursor = conn.cursor()

        # 获取所有文档
        cursor.execute("SELECT * FROM research_papers")
        papers = cursor.fetchall()

        # 准备批量操作
        actions = []
        for paper in papers:
            action = {
                "_index": index_name,
                "_id": paper[0],  # 使用SQLite的ID作为Elasticsearch的ID
                "_source": {
                    "title": paper[1],
                    "abstract": paper[2],
                    "authors": paper[3],
                    "publish_date": paper[4],
                    "category": paper[5],
                    "citations": paper[6],
                    "keywords": paper[7],
                    "created_at": paper[8]
                }
            }
            actions.append(action)

        # 批量导入
        success, _ = bulk(es, actions)
        print(f"成功导入 {success} 条文档到Elasticsearch")

        # 刷新索引使文档可搜索
        es.indices.refresh(index=index_name)
        conn.close()


    # 执行导入
    index_papers_to_es()


    # 简单全文搜索
    def search_papers(query):
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "abstract^2", "keywords"]  # 标题权重最高
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"citations": {"order": "desc"}}
            ],
            "highlight": {
                "fields": {
                    "title": {},
                    "abstract": {}
                }
            }
        }

        response = es.search(index=index_name, body=search_body)

        print(f"找到 {response['hits']['total']['value']} 条结果:")
        for hit in response['hits']['hits']:
            print(f"\nID: {hit['_id']}")
            print(f"标题: {hit['_source']['title']}")
            print(f"类别: {hit['_source']['category']}")
            print(f"引用量: {hit['_source']['citations']}")
            print(f"得分: {hit['_score']}")

            # 显示高亮内容
            if 'highlight' in hit:
                if 'title' in hit['highlight']:
                    print(f"高亮标题: {''.join(hit['highlight']['title'])}")
                if 'abstract' in hit['highlight']:
                    print(f"高亮摘要: {''.join(hit['highlight']['abstract'])}")

            print("-" * 80)


    # 执行搜索
    print("=== 搜索 'transformer' ===")
    search_papers("transformer")

    print("\n=== 搜索 'neural network' ===")
    search_papers("neural network")


    # 按类别筛选搜索
    def search_by_category(query, category):
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "abstract^2", "keywords"]
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"category": category}}
                    ]
                }
            },
            "sort": [
                {"citations": {"order": "desc"}}
            ]
        }

        response = es.search(index=index_name, body=search_body)

        print(f"在 '{category}' 类别中找到 {response['hits']['total']['value']} 条结果:")
        for hit in response['hits']['hits']:
            print(f"标题: {hit['_source']['title']}")
            print(f"引用量: {hit['_source']['citations']}")
            print(f"摘要片段: {hit['_source']['abstract'][:100]}...")
            print("-" * 80)


    print("\n=== 在'Computer Vision'类别中搜索 'image' ===")
    search_by_category("image", "Computer Vision")


    # 聚合查询 - 按类别统计论文数量
    def aggregate_by_category():
        search_body = {
            "size": 0,
            "aggs": {
                "papers_by_category": {
                    "terms": {
                        "field": "category",
                        "size": 10
                    }
                }
            }
        }

        response = es.search(index=index_name, body=search_body)

        print("=== 按类别统计论文数量 ===")
        for bucket in response['aggregations']['papers_by_category']['buckets']:
            print(f"{bucket['key']}: {bucket['doc_count']} 篇论文")


    aggregate_by_category()
