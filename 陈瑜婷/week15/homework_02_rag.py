# -*- coding: utf-8 -*-
"""
使用云端Qwen接口 + 云端MinerU解析PDF，并基于解析结果做RAG问答。
"""
import os
import json
import uuid
import time
import zipfile
import tempfile
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MINERU_API_KEY = os.getenv("MINERU_API_KEY")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# MinerU云端解析接口
# 获取上传URL的接口
MINERU_BATCH_URL = "https://mineru.net/api/v4/file-urls/batch"
# 查询解析结果的接口
MINERU_RESULT_URL = "https://mineru.net/api/v4/extract-results/batch"

EMBEDDING_MODEL = "text-embedding-v3"
CHAT_MODEL = "qwen-plus"


def parse_pdf_with_mineru(pdf_path: str, output_format: str = "markdown", model_version: str = "pipeline") -> Dict[str, Any]:
    """
    使用云端MinerU解析PDF，返回结构化内容。
    新接口流程：1) 获取上传URL 2) 上传文件 3) 查询解析结果
    :param pdf_path: 待解析PDF文件路径
    :param output_format: 期望输出格式，默认markdown，可根据接口能力调整
    :param model_version: 模型版本，默认pipeline
    """
   
    token = MINERU_API_KEY
    file_name = os.path.basename(pdf_path)
    
    # 生成data_id（使用uuid）
    data_id = str(uuid.uuid4())
    
    # 1. 获取上传URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": file_name, "data_id": data_id}
        ],
        "model_version": model_version
    }
    
    print(f"正在获取上传URL...")
    response = requests.post(MINERU_BATCH_URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()
    
    if result.get("code") != 0:
        raise ValueError(f"获取上传URL失败: {result.get('msg', '未知错误')}")
    
    batch_id = result["data"]["batch_id"]
    urls = result["data"]["file_urls"]
    
    if not urls or len(urls) == 0:
        raise ValueError("未获取到上传URL")
    
    upload_url = urls[0]
    print(f"获取到上传URL，batch_id: {batch_id}")
    
    # 2. 上传文件
    print(f"正在上传文件: {file_name}...")
    with open(pdf_path, 'rb') as f:
        res_upload = requests.put(upload_url, data=f, timeout=600)
        if res_upload.status_code != 200:
            raise ValueError(f"文件上传失败: HTTP {res_upload.status_code}, {res_upload.text}")
    
    print(f"文件上传成功")
    
    # 3. 查询解析结果（轮询直到完成）
    print(f"正在等待解析完成...")
    max_wait_time = 10 * 60  # 最大等待10分钟
    check_interval = 5  # 每5秒查询一次
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        query_url = f"{MINERU_RESULT_URL}/{batch_id}"
        query_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        try:
            query_resp = requests.get(query_url, headers=query_headers, timeout=30)
            query_resp.raise_for_status()
            query_result = query_resp.json()
        except Exception as e:
            print(f"查询解析结果时出错: {e}")
            time.sleep(check_interval)
            continue
        
        # 根据实际API返回结构调整
        if query_result.get("code") == 0:
            data = query_result.get("data", {})
            extract_results = data.get("extract_result", [])
            
            if not extract_results:
                print("警告: extract_result 为空，继续等待...")
                time.sleep(check_interval)
                continue
            
            # 获取第一个结果（通常只有一个文件）
            result_item = extract_results[0]
            state = result_item.get("state", "")
            
            print(f"当前状态: {state}")
            
            if state == "done":
                # 解析完成，下载zip文件并提取full.md
                full_zip_url = result_item.get("full_zip_url", "")
                if not full_zip_url:
                    raise ValueError("解析完成但未找到full_zip_url")
                
                print(f"正在下载解析结果: {full_zip_url}")
                
                # 下载zip文件
                zip_resp = requests.get(full_zip_url, timeout=300)
                zip_resp.raise_for_status()
                
                # 创建临时目录并解压
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_path = os.path.join(temp_dir, "result.zip")
                    with open(zip_path, "wb") as f:
                        f.write(zip_resp.content)
                    
                    # 解压zip文件
                    print("正在解压文件...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # 查找full.md文件
                    full_md_path = os.path.join(temp_dir, "full.md")
                    if not os.path.exists(full_md_path):
                        # 可能在子目录中
                        for root, dirs, files in os.walk(temp_dir):
                            if "full.md" in files:
                                full_md_path = os.path.join(root, "full.md")
                                break
                    
                    if not os.path.exists(full_md_path):
                        raise FileNotFoundError(f"解压后未找到full.md文件，解压目录内容: {os.listdir(temp_dir)}")
                    
                    # 读取full.md内容
                    with open(full_md_path, "r", encoding="utf-8") as f:
                        markdown_text = f.read()
                    
                    print(f"解析完成！已提取 {len(markdown_text)} 个字符")
                    return {
                        "output": markdown_text,
                        "batch_id": batch_id,
                        "request_id": batch_id
                    }
                    
            elif state == "failed":
                err_msg = result_item.get("err_msg", "未知错误")
                raise ValueError(f"解析失败: {err_msg}")
            elif state in ["pending", "running", "converting"]:
                # 继续等待
                pass
            else:
                print(f"未知状态: {state}，继续等待...")
        
        # 如果还在处理中，等待后继续查询
        time.sleep(check_interval)
        print(f"解析中，等待中... (已等待 {int(time.time() - start_time)} 秒)")
    
    raise TimeoutError(f"解析超时，超过 {max_wait_time} 秒未完成")


def split_markdown(markdown_text: str, max_chars: int = 800) -> List[str]:
    """
    将Markdown文本按长度切分，避免过长上下文影响召回质量。
    :param markdown_text: MinerU解析得到的Markdown文本
    :param max_chars: 单段最大字符数
    """
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0
    for line in markdown_text.splitlines():
        # 避免空行造成冗余
        if not line.strip():
            line = ""
        buffer.append(line)
        current_len += len(line)
        if current_len >= max_chars:
            chunks.append("\n".join(buffer).strip())
            buffer = []
            current_len = 0
    if buffer:
        chunks.append("\n".join(buffer).strip())
    # 过滤空段
    return [c for c in chunks if c]


class SimpleRAG:
    """简单RAG实现，复用Qwen Embedding + Chat。"""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, chat_model: str = CHAT_MODEL):
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=QWEN_BASE_URL)
        self.df: pd.DataFrame = pd.DataFrame()
        self.embeddings: np.ndarray | None = None

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本向量。"""
        resp = self.client.embeddings.create(model=self.embedding_model, input=text)
        return resp.data[0].embedding

    def build_index(self, chunks: List[str]) -> None:
        """为解析段落构建向量索引。"""
        rows = []
        vectors = []
        for idx, chunk in enumerate(chunks):
            emb = self._get_embedding(chunk)
            rows.append({"id": idx, "content": chunk})
            vectors.append(emb)
        self.df = pd.DataFrame(rows)
        self.embeddings = np.array(vectors)

    def retrieve(self, query: str, top_k: int = 4) -> pd.DataFrame:
        """基于余弦相似度召回文档段落。"""
        if self.embeddings is None or self.df.empty:
            raise RuntimeError("索引未构建，请先调用 build_index。")
        q_emb = np.array(self._get_embedding(query)).reshape(1, -1)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        result = self.df.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result

    def answer(self, query: str, top_k: int = 4) -> str:
        """组合检索上下文并生成回答。"""
        top_docs = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(
            [f"【片段{row['id']}|相似度:{row['similarity']:.4f}】\n{row['content']}" for _, row in top_docs.iterrows()]
        )
        messages = [
            {
                "role": "system",
                "content": "你是专业的文档问答助手，请严格基于参考内容回答，无法找到信息时需明确说明。",
            },
            {
                "role": "user",
                "content": f"参考内容：\n{context}\n\n问题：{query}\n请用中文回答。",
            },
        ]
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content


def run_once(pdf_path: str, question: str) -> str:
    """
    完整流程：解析PDF -> 切分 -> 建索引 -> RAG问答。
    :param pdf_path: PDF文件路径
    :param question: 用户问题
    """
    # 1) MinerU解析
    mineru_result = parse_pdf_with_mineru(pdf_path, output_format="markdown")
    markdown_text = mineru_result.get("output", "")

    # 2) 切分
    chunks = split_markdown(markdown_text, max_chars=800)

    # 3) 建索引 + 问答
    rag = SimpleRAG()
    rag.build_index(chunks)
    return rag.answer(question)


if __name__ == "__main__":
    
    default_pdf = os.path.join(
        os.path.dirname(__file__),
        "合合信息2025年AIOCR智能审核实战手册45页.pdf",
    )
    if not os.path.exists(default_pdf):
        raise FileNotFoundError(f"未找到示例PDF: {default_pdf}")

    user_question = input("请输入你的问题：").strip()
    if not user_question:
        raise ValueError("问题不能为空。")

    answer = run_once(default_pdf, user_question)
    print("\n===== 回答 =====")
    print(answer)

