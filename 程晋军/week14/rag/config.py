import os
# from dotenv import load_dotenv

# load_dotenv()  # 加载.env文件中的密钥

# 千问Embedding配置（二选一：API版/开源本地版）
QWEN_EMBEDDING_CONFIG = {
    "use_api": True,  # True用API，False用本地开源模型
    "api_key": "sk-8dfd0034a9d7404b827dad9b02e1e9d4",  # 阿里云DashScope密钥
    "local_model_name": "qwen/Qwen-Embedding-V2",  # 开源模型名称
    "embedding_dim": 1024,  # 向量维度（V2版为768，0.6B版为384）
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
}

# 文本处理配置
TEXT_SPLIT_CONFIG = {
    "chunk_size": 800,  # 文本块长度
    "chunk_overlap": 100  # 块间重叠长度
}

# 向量库配置
VECTOR_DB_CONFIG = {
    "db_path": "./qwen_embedding_db",  # 向量库存储路径
    "top_k": 5  # 检索匹配top数量
}