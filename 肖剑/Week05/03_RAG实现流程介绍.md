1.  **RAG系统概述**&#x20;

    本系统实现了一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的政企问答解决方案。系统通过结合文档检索和大语言模型生成能力，为用户提供基于知识库内容的准确回答。

2.  **系统架构**

    2.1 核心模块

    *   **文档处理模块**：负责解析和存储各类文档

    *   **向量编码模块**：将文本转换为向量表示

    *   **检索模块**：实现多路混合检索

    *   **重排序模块**：对检索结果进行精排

    *   **生成模块**：基于检索内容生成回答

    2.2 相关技术

    *   向量存储：Elasticsearch
    *   嵌入模型：BGE系列（bge-small-zh-v1.5/bge-base-zh-v1.5）
    *   重排序模型：bge-reranker-base
    *   大语言模型：OpenAI API兼容模型
    *   文档解析：pdfplumber（PDF处理）

3.  **工作流程**

    3.1 初始化阶段

    *   加载配置文件(config.yaml)
    *   根据配置初始化嵌入模型（load\_embdding\_model）和重排序模型（load\_rerank\_model）
    *   将模型加载到指定设备（CPU/GPU）
    *   创建RAG类实例，设置相关参数

    3.2 文档处理流程

    *   文档解析（使用pdfplumber打开PDF文件；逐页提取文本内容；前3页内容作为文档摘要）
    *   文本处理与存储（存储每页文本内容；使用重叠分块策略将每页内容划分chunk；为每个文本块生成向量表示；将文本块和向量存储到Elasticsearch）
    *   元数据存储（存储文档基本信息；保存文档摘要和文件路径）

    3.3 查询处理流程

    ```python
    def query_document(self, query: str, knowledge_id: int) -> List[str]:
    ```

    *   多路检索（全文检索：BM25 和 语义检索：KNN）
    *   结果融合（使用rrf算法融合两种检索结果；公式：score = 1/(rank + k)，其中k=60）
    *   重排序

    3.4 对话生成流程

    ```python
    # 回答生成主要步骤
    def chat_with_rag(knowledge_id: int, messages: List[Dict]):
        if len(messages) == 1:  # 第一轮对话
            # 1. 检索相关文档
            # 2. 构建提示词
            # 3. 调用LLM生成回答
        else:  # 多轮对话
            # 直接使用LLM基于对话历史生成回答
    ```

    *   提示词构建（使用预定义模版组合当前时间、检索内容和用户问题）
    *   LLM调用（调用配置的LLM API生成回答；支持调节temperature和top\_p参数）
    *   对话管理（第一轮对话使用RAG模式（检索+生成）；后续对话直接使用LLM而不检索）

4.  **核心算法与实现**

    4.1 文本分块算法

    *   使用固定大小的滑动窗口进行文本分块
    *   支持设置重叠区域，避免重要信息被切割

    ```python
    def split_text_with_overlap(text, chunk_size, chunk_overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = start + chunk_size - chunk_overlap        
        return chunks
    ```

    4.2 混合检索与结果融合

    *   采用RRF算法融合全文检索和语义检索结果
    *   兼顾关键词匹配和语义相似度&#x20;

    4.3 重排序算法
