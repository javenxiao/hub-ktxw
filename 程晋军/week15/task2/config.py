import os
from dotenv import load_dotenv

load_dotenv()

# 替换为OpenAI配置（二选一：API版/本地开源版不变，仅API提供商切换）
EMBEDDING_CONFIG = {
    "use_api": True,  # True用OpenAI API，False保留本地开源模型
    "openai_api_key": "sk-67a5c9ec491c4832b45b0cbb567bc67f",
    "openai_embedding_model": "text-embedding-v4",  # 轻量高效，可选text-embedding-3-large
    "local_model_name": "qwen/Qwen-Embedding-V2",
    "embedding_dim": 1024,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
}

TEXT_SPLIT_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100
}

VECTOR_DB_CONFIG = {
    "db_path": "./openai_embedding_db",
    "top_k": 5
}

# OpenAI大模型配置
OPENAI_LLM_CONFIG = {
    "model": "qwen-flash",  # 性价比首选，可选gpt-4o
    "temperature": 0.3
}