import yaml  # type: ignore
from elasticsearch import Elasticsearch  # type: ignore  # ES的Python客户端（用于操作ES）
import traceback  # 用于捕获和打印详细的错误信息（方便排查问题）
from pathlib import Path  # 导入 pathlib

CONFIG_PATH = Path(__file__).parent / "config.yaml"


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

"""
初始化ES连接并创建符合RAG需求的索引结构，确保后续能正常存储和检索数据
"""

# 从配置文件中提取ES的连接参数
es_host = config["elasticsearch"]["host"]          # ES服务器的主机地址
es_port = config["elasticsearch"]["port"]          # ES的端口号（默认9200）
es_scheme = config["elasticsearch"]["scheme"]      # 连接协议（通常是"http"或"https"）
es_username = config["elasticsearch"]["username"]  # 认证用户名
es_password = config["elasticsearch"]["password"]  # 认证密码

# 根据是否有用户名密码，创建不同的ES客户端
if es_username != "" and es_password != "":
    # 有认证的情况：使用账号密码连接
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
        basic_auth=(es_username, es_password)  # 传入认证信息
    )
else:
    # 无认证的情况：直接连接（适合本地测试环境）
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
    )

# 从配置中获取嵌入向量的维度（与嵌入模型匹配）
embedding_dims = config["models"]["embedding_model"][
    config["rag"]["embedding_model"]  # 从rag配置中获取当前使用的嵌入模型
]["dims"]  # 对应模型的向量维度


# 初始化ES环境
def init_es():
    """
    检查ES连接并创建必要的索引结构
    :return: 环境是否配置成功（True/False）
    """

    if not es.ping():  # ping()方法测试与ES的连接
        print("Could not connect to Elasticsearch.")
        return False

    document_meta_mapping = {
        "mappings": {    # mappings定义索引的字段结构
            'properties': {    # 定义索引中的所有字段
                'file_name': {      # 字段1：文件名
                    'type': 'text',  # 文本类型（可分词检索）
                    'analyzer': 'ik_max_word',   # 索引时分词器（ik_max_word：中文细粒度分词，适合长文本）
                    'search_analyzer': 'ik_max_word'   # 查询时分词器（与索引时保持一致，确保检索准确）
                },
                'abstract': {       # 字段2：文档摘要（简要描述文档内容）
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                },
                'full_content': {   # 字段3：文档完整内容（可选，大文档可能不存储）
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                }
            }
        }
    }
    try:
        # 检查索引是否已存在，不存在则创建
        # es.indices.delete(index='document_meta')
        if not es.indices.exists(index="document_meta"):   # 检查索引是否存在
            es.indices.create(index='document_meta', body=document_meta_mapping)  # 创建索引
    except:
        print(traceback.format_exc())  # 打印异常详情
        print("Could not create index of document_meta.")
        return False

    chunk_info_mapping = {
        'mappings': {  # Add 'mappings' here  文档分块内容（如将PDF按256token拆分后的片段）
            'properties': {
                'chunk_content': {     # 字段1：文档分块内容
                    'type': 'text',
                    'analyzer': 'ik_max_word',
                    'search_analyzer': 'ik_max_word'
                },
                "embedding_vector": { 
                    "type": "dense_vector",  
                    "element_type": "float",  
                    "dims": embedding_dims, 
                    "index": True,           
                    "index_options": {
                        "type": "int8_hnsw" 
                    }
                }
            }
        }
    }

    try:
        # 检查索引是否存在，不存在则创建
        # es.indices.delete(index='chunk_info')  # 开发时删除旧索引用
        if not es.indices.exists(index="chunk_info"):
            es.indices.create(index='chunk_info', body=chunk_info_mapping)
    except:
        print(traceback.format_exc())
        print("Could not create index of chunk_info.")
        return False

    print("Successfully connected to Elasticsearch!")
    return True


init_es()
