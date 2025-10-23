from elasticsearch import Elasticsearch
import json
import time
from sentence_transformers import SentenceTransformer

# ---------------- 初始化 ----------------
es = Elasticsearch("http://localhost:9200")

if es.ping():
    print("成功连接到 Elasticsearch！")
else:
    print("无法连接到 Elasticsearch，请检查服务是否运行。")

students = [
    {
        "student_id": "S001",
        "name": "张三",
        "gender": "男",
        "age": 20,
        "major": "计算机科学",
        "score": 92.5,
        "introduction": "张三在人工智能方向表现突出，积极参与深度学习相关的科研项目。"
    },
    {
        "student_id": "S002",
        "name": "李四",
        "gender": "女",
        "age": 21,
        "major": "数学",
        "score": 88.0,
        "introduction": "李四对数据分析有浓厚兴趣，擅长数学建模，参加多次竞赛获奖。"
    },
    {
        "student_id": "S003",
        "name": "王五",
        "gender": "男",
        "age": 22,
        "major": "物理",
        "score": 75.0,
        "introduction": "王五在量子物理方向具有潜力，参与实验室项目，发表学术论文。"
    },
    {
        "student_id": "S004",
        "name": "赵六",
        "gender": "女",
        "age": 20,
        "major": "计算机科学",
        "score": 95.0,
        "introduction": "赵六在自然语言处理方向成绩优异，精通机器学习和深度学习。"
    }
]

# ---------------- 传统关键词搜索 ----------------
print("\n========== 传统关键词搜索 ==========")

index_traditional = "student_search_traditional"

# 删除旧索引
if es.indices.exists(index=index_traditional):
    es.indices.delete(index=index_traditional)
    print(f"旧索引 '{index_traditional}' 已删除。")
    time.sleep(1)

# 创建新索引
es.indices.create(
    index=index_traditional,
    body={
        "mappings": {
            "properties": {
                "student_id": {"type": "keyword"},
                "name": {"type": "text", "analyzer": "ik_max_word"},
                "gender": {"type": "keyword"},
                "age": {"type": "integer"},
                "major": {"type": "text", "analyzer": "ik_max_word"},
                "score": {"type": "float"},
                "introduction": {"type": "text", "analyzer": "ik_smart"}
            }
        }
    }
)
print(f"索引 '{index_traditional}' 创建成功。")

# 插入数据
for stu in students:
    es.index(index=index_traditional, document=stu)
es.indices.refresh(index=index_traditional)

# 查询：专业是计算机的学生，成绩 > 85
res_1 = es.search(
    index=index_traditional,
    body={
        "query": {
            "bool": {
                "must": {
                    "match": {"major": "计算机"}
                },
                "filter": [
                    {"range": {"score": {"gt": 85}}}
                ]
            }
        }
    }
)

print("\n--- 查询结果（计算机专业，成绩 > 85） ---")
for hit in res_1['hits']['hits']:
    print(f"得分：{hit['_score']}\n{json.dumps(hit['_source'], ensure_ascii=False, indent=2)}")

# ---------------- 语义搜索 ----------------
print("\n========== 语义搜索 ==========")
print("正在加载 SentenceTransformer 模型...")
model = SentenceTransformer('../models/BAAI/bge-small-zh-v1.5/')
print("模型加载完成。")

index_semantic = "student_semantic_search"

# 删除旧索引
if es.indices.exists(index=index_semantic):
    es.indices.delete(index=index_semantic)
    print(f"旧索引 '{index_semantic}' 已删除。")
    time.sleep(1)

# 创建新索引（向量字段）
es.indices.create(
    index=index_semantic,
    body={
        "mappings": {
            "properties": {
                "name": {"type": "text", "analyzer": "ik_max_word"},
                "major": {"type": "text", "analyzer": "ik_max_word"},
                "introduction": {"type": "text", "analyzer": "ik_smart"},
                "vector": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
)
print(f"索引 '{index_semantic}' 创建成功。")

# 插入学生数据（带向量）
for stu in students:
    text_to_encode = stu["introduction"]
    vector = model.encode(text_to_encode).tolist()
    es.index(
        index=index_semantic,
        document={
            "name": stu["name"],
            "major": stu["major"],
            "introduction": stu["introduction"],
            "vector": vector
        }
    )
print("所有学生数据插入完成。")
es.indices.refresh(index=index_semantic)

# 语义搜索
queries = [
    "学习人工智能的优秀学生",
    "对数学和建模很有兴趣的人",
    "研究自然语言处理方向的学生"
]

for q in queries:
    q_vector = model.encode(q).tolist()
    response = es.search(
        index=index_semantic,
        body={
            "knn": {
                "field": "vector",
                "query_vector": q_vector,
                "k": 2,
                "num_candidates": 5
            },
            "fields": ["name", "major", "introduction"],
            "_source": False
        }
    )
    print(f"\n查询: '{q}'")
    for hit in response['hits']['hits']:
        print(f"得分: {hit['_score']:.4f}, 姓名: {hit['fields']['name'][0]}, 专业: {hit['fields']['major'][0]}, 简介: {hit['fields']['introduction'][0]}")
