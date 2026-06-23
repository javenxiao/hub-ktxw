# -*- coding: utf-8 -*-
"""
基于RAG的问答调用公式计算结果的功能
使用云端Qwen系列模型进行embedding和agent推理
"""
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from agents.mcp.server import MCPServerSse
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.mcp import MCPServer, ToolFilterStatic
from agents import set_default_openai_api, set_tracing_disabled
from typing import List, Dict, Any, Optional
from config import DOCUMENTS_CSV_PATH, EMBEDDING_MODEL, LLM_MODEL

# 设置API配置
# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class RAGKnowledgeBase:
    """RAG知识库类，用于加载文档并进行向量检索"""
    
    def __init__(self, csv_path: str, embedding_model: str = "text-embedding-v3"):
        """
        初始化RAG知识库
        :param csv_path: 文档CSV文件路径
        :param embedding_model: embedding模型名称
        """
        self.csv_path = csv_path
        self.embedding_model = embedding_model
        self.documents_df = None
        self.embeddings = None
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"]
        )
        print("正在加载知识库...")
        self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """加载知识库CSV文件"""
        try:
            self.documents_df = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"成功加载 {len(self.documents_df)} 条文档记录")
            
            # 生成所有文档的embeddings
            print("正在生成文档embeddings...")
            self.generate_embeddings()
            print("知识库加载完成！")
        except Exception as e:
            print(f"加载知识库失败: {e}")
            raise
    
    def get_embedding(self, text: str) -> list:
        """
        获取文本的embedding向量
        :param text: 输入文本
        :return: embedding向量
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取embedding失败: {e}")
            return None
    
    def generate_embeddings(self):
        """为所有文档生成embedding向量"""
        embeddings_list = []
        embedding_dim = None
        
        for idx, row in self.documents_df.iterrows():
            content = row['content']
            embedding = self.get_embedding(content)
            if embedding:
                embeddings_list.append(embedding)
                # 记录embedding维度
                if embedding_dim is None:
                    embedding_dim = len(embedding)
            else:
                # 如果失败，添加零向量
                if embedding_dim is None:
                    embedding_dim = 1536  # text-embedding-v3的默认维度
                embeddings_list.append([0.0] * embedding_dim)
            
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{len(self.documents_df)} 条文档")
        
        self.embeddings = np.array(embeddings_list)
        print(f"Embedding维度: {embedding_dim}")
    
    def retrieve_top_k(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        根据查询检索最相关的top_k个文档
        :param query: 查询文本
        :param top_k: 返回的文档数量
        :return: 包含top_k相关文档的DataFrame
        """
        # 获取查询的embedding
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            print("查询embedding生成失败")
            return pd.DataFrame()
        
        # 计算余弦相似度
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # 获取top_k索引
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 返回top_k文档
        top_k_docs = self.documents_df.iloc[top_k_indices].copy()
        top_k_docs['similarity'] = similarities[top_k_indices]
        
        return top_k_docs


async def run_rag_qa(rag_kb: RAGKnowledgeBase):
    """
    运行基于RAG的问答系统
    :param rag_kb: RAG知识库实例
    """
    # 创建OpenAI客户端
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    
    # 交互式问答循环
    print("\n" + "="*60)
    print("基于RAG的公式计算问答系统已启动！")
    print("输入问题进行查询，输入 'exit' 或 'quit' 退出")
    print("="*60 + "\n")
    
    while True:
        # 获取用户输入
        user_query = input("\n请输入您的问题: ").strip()
        
        if not user_query:
            continue
        
        if user_query.lower() in ['exit', 'quit', '退出']:
            print("感谢使用，再见！")
            break
        
        try:
            print(f"\n正在检索相关文档...")
            
            # 1. 使用RAG检索top5相关文档
            top_k_docs = rag_kb.retrieve_top_k(user_query, top_k=5)
            
            if top_k_docs.empty:
                print("未找到相关文档，请重新输入问题。")
                continue
            
            # 2. 提取相关的formula作为tool_filter
            relevant_formulas = top_k_docs['formula'].unique().tolist()
            
            print(f"找到 {len(top_k_docs)} 条相关文档")
            print(f"相关公式工具: {', '.join(relevant_formulas)}")
            
            # 3. 构建上下文信息
            context = "\n\n".join([
                f"【文档 {i+1}】(相似度: {row['similarity']:.4f})\n{row['content'][:500]}..."
                for i, (idx, row) in enumerate(top_k_docs.iterrows())
            ])
            
            # 4. 构建增强的prompt
            enhanced_prompt = f"""基于以下参考文档回答用户问题。如果需要进行计算，请使用提供的工具函数。

参考文档：
{context}

用户问题：{user_query}

请基于参考文档和工具函数给出详细的回答。"""
            
            # 5. 创建带有tool_filter的MCP服务器
            # mcp tools 选择
            if not relevant_formulas or len(relevant_formulas) == 0:
                tool_mcp_tools_filter: Optional[ToolFilterStatic] = None
            else:
                tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=relevant_formulas)
            
            async with MCPServerSse(
                name="Formulas MCP Server",
                params={
                    "url": "http://localhost:8900/sse",
                },
                tool_filter=tool_mcp_tools_filter,
                cache_tools_list=False,
                client_session_timeout_seconds=20,
            ) as filtered_server:
                
                # 6. 创建Agent
                instructions = f"""你是一个专业的数学建模和公式计算助手。请基于提供的参考文档和工具函数，准确回答用户的问题。
                
当前可用的工具函数：{', '.join(relevant_formulas)}

如果问题需要计算，请优先使用这些工具函数。"""
                
                agent = Agent(
                    name="RAG计算助手",
                    instructions=instructions,
                    mcp_servers=[filtered_server],
                    model=OpenAIChatCompletionsModel(
                        model=LLM_MODEL,
                        openai_client=external_client,
                    )
                )
                
                # 7. 执行推理
                print(f"\n正在生成回答...")
                result = await Runner.run(starting_agent=agent, input=enhanced_prompt)
                
                # 8. 输出结果
                print("\n" + "="*60)
                print("回答:")
                print("="*60)
                print(result.final_output)
                print("="*60)
            
        except Exception as e:
            print(f"\n处理问题时发生错误: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    try:
        # 1. 初始化RAG知识库
        rag_kb = RAGKnowledgeBase(
            csv_path=str(DOCUMENTS_CSV_PATH),
            embedding_model=EMBEDDING_MODEL
        )
        
        # 2. 运行RAG问答系统（每次查询时动态创建带tool_filter的MCP连接）
        await run_rag_qa(rag_kb)
            
    except Exception as e:
        print(f"系统启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())