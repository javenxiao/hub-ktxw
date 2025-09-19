import traceback

from elasticsearch import Elasticsearch
from my_data import DATAS

# 索引名称
INDEX_NAME = 'example'

# 索引映射
INDEX_MAPPING = {
    'settings': {
        'number_of_shards': 1,
        'number_of_replicas': 0
    },
    'mappings': {
        'properties': {
            'doc_id': {
                'type': 'keyword'
            },
            'title': {
                'type': 'text',
                'analyzer': 'ik_max_word',
                'search_analyzer': 'ik_smart',
                'fields': {
                    'keyword': {
                        'type': 'keyword',
                        'ignore_above': 256
                    }
                }
            },
            'content': {
                'type': 'text',
                'analyzer': 'ik_max_word',
                'search_analyzer': 'ik_smart'
            },
            'category': {
                'type': 'keyword'
            },
            'author': {
                'type': 'keyword'
            },
            'publish_date': {
                'type': 'date',
                'format': 'yyyy-MM-dd'  # 日期格式
            },
            'views': {
                'type': 'integer'
            },
            'tags': {
                'type': 'keyword'
            },
            'is_recommended': {
                'type': 'boolean'
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

# 添加数据
try:
    for doc in DATAS:
        es.index(index=INDEX_NAME, document=doc, id=doc['doc_id'])
        print('Elasticsearch添加文档成功 -> ', doc['doc_id'])
except Exception as e:
    print('Elasticsearch添加文档失败: ', e)
    traceback.print_exc()

# 刷新
es.indices.refresh(index=INDEX_NAME)
print('Elasticsearch添加文档后刷新')


def search(query):
    """
    Elasticsearch search
    :param query:
    :return:
    """
    result = es.search(index=INDEX_NAME, body=query)
    return result['hits']['hits']

# 搜索1：标题检索
query1 = {
    'query': {
        'match': {
            'title': '搜索'
        }
    }
}
result1 = search(query1)
print('全文检索标题有"搜索"的文档结果：')
for doc in result1:
    print(doc)

# 搜索2：标题和内容检索
query2 = {
    'query': {
        'multi_match': {
            'query': '搜索',
            'fields': ['title', 'content']
        }
    }
}
result2 = search(query2)
print('全文检索标题或者内容有"搜索"的文档结果：')
for doc in result2:
    print(doc)

# 搜索3：精确匹配
query3 = {
    'query': {
        'term': {
            'category': '技术文档'
        }
    }
}
result3 = search(query3)
print('精确匹配类型是"技术文档"的文档结果：')
for doc in result3:
    print(doc)


# 搜索4：组合搜索
query4 = {
    'query': {
        'bool': {
            'must': [
                {
                    'term': {
                        'category': '技术文档'
                    }
                }
            ],
            'filter': [
                {
                    'term': {
                        'author': '张三'
                    }
                }
            ]
        }
    }
}
result4 = search(query4)
print('组合搜索类型是"技术文档"而且作者是"张三"的文档结果：')
for doc in result4:
    print(doc)