import logging
from typing import List, Dict, Any, Tuple
from openai import OpenAI


class RAGEngine:
    """RAG引擎模块 - 检索增强生成核心逻辑[9,10](@ref)"""

    def __init__(self, api_key: str = None, model_name: str = "qwen-max"):
        self.api_key = "sk-2ff484a65dbd47668f71c459353fd8ff" if api_key is None else api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.logger = logging.getLogger(__name__)

    def _initialize_llm(self):
        """初始化语言模型"""
        if not self.api_key:
            self.logger.warning("未提供API密钥，将尝试使用本地模型")
            # 这里可以扩展为使用本地模型
            return None

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def chat(self, messages: List[Dict], top_p: float = 10, temperature: float = 0.2) -> Any:
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message

    def build_context(self, similar_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
        """构建检索上下文"""
        context_parts = []
        for chunk, score in similar_chunks:
            context_parts.append(f"[相关度: {score:.3f}] {chunk['content']}")

        return "\n\n".join(context_parts)

    def generate_answer(self, question: str, context: str) -> str:
        """
        生成答案[9,10](@ref)

        Args:
            question: 用户问题
            context: 检索到的上下文

        Returns:
            生成的答案
        """
        prompt_template = """基于以下检索到的文档内容，请回答问题。如果内容不相关或信息不足，请如实说明。

检索到的相关内容：
{context}

问题：{question}

请根据以上信息提供准确、简洁的回答："""

        prompt = prompt_template.format(context=context, question=question)

        if self.llm is None:
            return f"LLM未正确初始化。问题: {question}\n检索到的上下文: {context[:500]}..."

        try:
            messages = [
                {"role": "system", "content": "你是一个专业的文档问答助手，基于提供的文档内容准确回答问题。"},
                {"role": "user", "content": prompt}
            ]

            response = self.chat(messages)
            return response

        except Exception as e:
            self.logger.error(f"答案生成失败: {e}")
            return f"无法生成答案，错误: {str(e)}"

    def rag_query(self, question: str, vector_indexer, k: int = 3) -> Dict[str, Any]:
        """
        执行RAG查询

        Args:
            question: 用户问题
            vector_indexer: 向量索引器实例
            k: 检索数量

        Returns:
            包含查询结果的字典
        """
        # 检索相似内容
        similar_chunks = vector_indexer.search_similar(question, k)

        if not similar_chunks:
            return {
                "question": question,
                "answer": "未找到相关文档内容",
                "sources": [],
                "retrieval_count": 0
            }

        # 构建上下文
        context = self.build_context(similar_chunks)

        # 生成答案
        answer = self.generate_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": [chunk for chunk, score in similar_chunks],
            "retrieval_count": len(similar_chunks),
            "retrieval_scores": [score for chunk, score in similar_chunks]
        }
