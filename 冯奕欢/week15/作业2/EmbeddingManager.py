from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class ChromaEmbeddingManager:

    def __init__(self, chroma_db_path="./chroma_rag_db"):
        self.chroma_db_path = chroma_db_path  # 向量库持久化路径
        self.embeddings = self._init_embeddings()  # 中文嵌入模型
        self.vector_db = None  # 初始化后赋值


    # --------------------------
    # 初始化中文嵌入模型（适配RAG）
    # --------------------------
    def _init_embeddings(self):
        """通用中文嵌入模型：BGE-small-zh-v1.5（轻量且精准）"""
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},  # 无GPU则用cpu，有GPU改为"cuda"
            encode_kwargs={"normalize_embeddings": True}  # 归一化提升检索精度
        )

    # --------------------------
    # 1. 将Chunk存入Chroma（RAG知识库）
    # --------------------------
    def save_chunks_to_chroma(self, all_chunks: list):
        """
        将所有Chunk存入Chroma向量库（持久化，重启后不丢失）
        关键：保留Chunk的元数据，支持按类型检索
        """
        # 初始化Chroma向量库（持久化到本地路径）
        self.vector_db = Chroma.from_documents(
            documents=all_chunks,  # 传入所有拆分的Chunk
            embedding=self.embeddings,  # 中文嵌入模型
            persist_directory=self.chroma_db_path,  # 本地持久化路径
            collection_name="mineru_rag_kb"  # 知识库名称（可自定义）
        )
        print(f"✅ RAG知识库已保存至：{self.chroma_db_path}")
        print(f"✅ 知识库中Chunk总数：{self.vector_db._collection.count()}")

    # --------------------------
    # 2. RAG检索功能（通用查询接口）
    # --------------------------
    def rag_retrieve(self, query, top_k=3, filter_type=None):
        """
        从Chroma知识库检索相关Chunk
        :param query: 用户查询问题（中文）
        :param top_k: 返回最相关的k个Chunk
        :param filter_type: 过滤检索类型（text/table，None则不过滤）
        :return: 检索结果列表
        """
        if not self.vector_db:
            # 若未初始化，先加载本地知识库
            self.vector_db = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings,
                collection_name="mineru_rag_kb"
            )

        # 构建过滤条件（可选：仅检索文本/表格Chunk）
        where_clause = None
        if filter_type in ["text", "table"]:
            where_clause = {"type": filter_type}

        # 检索相关Chunk
        retriever = self.vector_db.as_retriever(
            search_kwargs={
                "k": top_k,
                "filter": where_clause  # 过滤条件
            }
        )
        relevant_chunks = retriever.invoke(query)

        # 格式化输出结果
        results = []
        for idx, chunk in enumerate(relevant_chunks, 1):
            results.append({
                "rank": idx,
                "score": chunk.metadata.get("score", "N/A"),  # 相似度分数（部分版本需手动计算）
                "type": chunk.metadata["type"],
                "source": chunk.metadata["file_name"],
                "content": chunk.page_content
            })
        return results