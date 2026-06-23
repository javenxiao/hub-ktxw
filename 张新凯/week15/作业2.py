"""
基于MinerU解析与RAG的智能文档问答系统
支持PDF/Word等多种文档格式

本地安装MinerU:
    pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
    pip install uv -i https://mirrors.aliyun.com/pypi/simple
    uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple
其他需要安装的：
    pip install mineru-client  # MinerU客户端
    pip install langchain langchain-community  # RAG框架
    pip install chromadb sentence-transformers  # 向量数据库和嵌入模型
"""


import os
import json
import logging
from typing import List, Dict, Any
import tempfile

# 第三方库导入
from mineru import MineruClient  # MinerU文档解析
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MineruRAGSystem:
    """基于MinerU和RAG的文档问答系统"""

    def __init__(self,
                 mineru_api_key: str = None,
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 llm_model: str = "qwen-plus",
                 llm_api_key: str = None,
                 llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 vector_store_path: str = "./chroma_db"):
        """
        初始化RAG系统

        Args:
            mineru_api_key: MinerU API密钥，None表示使用本地部署
            embedding_model: 嵌入模型名称
            llm_model: 大语言模型名称
            llm_api_key: 大语言模型api key
            llm_base_url: 大语言模型api url
            vector_store_path: 向量数据库存储路径
        """
        self.mineru_api_key = mineru_api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.vector_store_path = vector_store_path

        # 初始化组件
        self.mineru_client = self._init_mineru_client()
        self.embeddings = self._init_embeddings()
        self.vector_store = None
        self.qa_chain = None

    def _init_mineru_client(self):
        """初始化MinerU客户端"""
        # 方式1: 使用云端API（需要申请API Key）
        if self.mineru_api_key:
            logger.info("使用MinerU云端API服务")
            return MineruClient(api_key=self.mineru_api_key)

        # 方式2: 本地部署（需要安装MinerU）
        logger.info("使用本地MinerU服务")
        # MinerU服务运行在本地http://localhost:8000
        return MineruClient(base_url="http://localhost:8000")

    def _init_embeddings(self):
        """初始化文本嵌入模型"""
        logger.info(f"加载嵌入模型: {self.embedding_model}")
        # 使用HuggingFace嵌入模型
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def parse_document(self, file_path: str, output_format: str = "markdown") -> str:
        """
        使用MinerU解析文档

        Args:
            file_path: 文档路径（支持PDF、Word、PPT等）
            output_format: 输出格式，支持"markdown"、"json"等

        Returns:
            解析后的文本内容
        """
        logger.info(f"开始解析文档: {file_path}")

        try:
            # 调用MinerU解析文档
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # 解析文档
            result = self.mineru_client.parse(
                file_data=file_data,
                file_name=os.path.basename(file_path),
                output_format=output_format
            )

            if output_format == "json":
                # 如果是JSON格式，提取文本内容
                content = self._extract_text_from_json(result)
            else:
                content = result

            logger.info(f"文档解析完成，长度: {len(content)} 字符")
            return content

        except Exception as e:
            logger.error(f"文档解析失败: {e}")
            raise

    def _extract_text_from_json(self, json_data: Dict) -> str:
        """从MinerU的JSON输出中提取文本"""
        text_parts = []

        # 根据MinerU的JSON结构提取文本
        if isinstance(json_data, dict):
            if 'pages' in json_data:
                for page in json_data['pages']:
                    if 'blocks' in page:
                        for block in page['blocks']:
                            if 'text' in block:
                                text_parts.append(block['text'])

        return '\n'.join(text_parts)

    def create_vector_store(self, documents: List[Document]):
        """
        创建向量数据库

        Args:
            documents: 分割后的文档列表
        """
        logger.info(f"创建向量数据库，文档数量: {len(documents)}")

        # 使用Chroma作为向量数据库
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )

        # 持久化存储
        self.vector_store.persist()
        logger.info(f"向量数据库已保存到: {self.vector_store_path}")

    def setup_qa_chain(self, chain_type: str = "stuff"):
        """
        设置问答链

        Args:
            chain_type: 链类型，可选"stuff"、"map_reduce"、"refine"等
        """
        if not self.vector_store:
            raise ValueError("请先创建向量数据库")

        # 创建检索器
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4}  # 返回前4个相关文档块
        )

        # 自定义提示模板
        prompt_template = """基于以下上下文信息回答问题。如果你不知道答案，就说不知道，不要编造信息。

上下文:
{context}

问题: {question}

请根据上下文提供准确、简洁的回答。如果上下文中有多个相关信息，请进行综合。

回答:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # 初始化LLM
        llm = ChatOpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            model=self.llm_model,
            temperature=0.1,  # 低温度以获得更确定性的回答
            max_tokens=500
        )

        # 创建问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        logger.info("问答链设置完成")

    def process_document(self, file_path: str):
        """
        完整处理文档：解析、分块、向量化

        Args:
            file_path: 文档路径
        """
        # 1. 使用MinerU解析文档
        logger.info("步骤1: 使用MinerU解析文档")
        document_text = self.parse_document(file_path, output_format="markdown")

        # 2. 文本分割
        logger.info("步骤2: 文本分割")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 每个块500字符
            chunk_overlap=50,  # 重叠50字符
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        # 创建LangChain文档对象
        documents = text_splitter.create_documents([document_text])

        # 添加元数据
        for i, doc in enumerate(documents):
            doc.metadata = {
                "source": file_path,
                "chunk_id": i,
                "total_chunks": len(documents)
            }

        # 3. 创建向量数据库
        logger.info("步骤3: 创建向量数据库")
        self.create_vector_store(documents)

        # 4. 设置问答链
        logger.info("步骤4: 设置问答链")
        self.setup_qa_chain()

        logger.info(f"文档处理完成，共处理 {len(documents)} 个文本块")

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        提问并获取答案

        Args:
            question: 用户问题

        Returns:
            包含答案和源文档的字典
        """
        if not self.qa_chain:
            raise ValueError("请先处理文档并设置问答链")

        logger.info(f"处理问题: {question}")

        try:
            # 执行问答
            result = self.qa_chain({"query": question})

            # 提取源文档信息
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "...",  # 截取前200字符
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)

            return {
                "question": question,
                "answer": result["result"],
                "sources": sources,
                "has_answer": len(result["result"].strip()) > 0
            }

        except Exception as e:
            logger.error(f"问答失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，回答问题出错: {str(e)}",
                "sources": [],
                "has_answer": False
            }

    def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量提问"""
        results = []
        for question in questions:
            result = self.ask_question(question)
            results.append(result)
        return results


# 使用示例
def main():
    """主函数示例"""
    # 初始化系统
    rag_system = MineruRAGSystem(
        mineru_api_key=None,  # 使用本地MinerU，如需云端服务请填写API Key
        embedding_model="BAAI/bge-small-zh-v1.5",
        llm_model="qwen-plus",
        llm_api_key="sk-xxx",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        vector_store_path="./my_doc_db"
    )

    # 文档路径
    doc_path = "2509-MinerU2.5.pdf"

    # 处理文档
    print("开始处理文档...")
    rag_system.process_document(doc_path)

    # 交互式问答
    print("\n文档处理完成！现在可以提问了。输入'quit'退出。")

    question = input("\n请输入你的问题: ").strip()

    result = rag_system.ask_question(question)

    print(f"\n答案: {result['answer']}")

    if result['sources']:
        print(f"\n参考来源 ({len(result['sources'])}个):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['content']}")


if __name__ == "__main__":
    main()
