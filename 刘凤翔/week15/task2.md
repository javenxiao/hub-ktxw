
### 一、核心流程（从解析到问答闭环）
#### 1. 用Mineru解析PDF/Word（关键第一步）
- 核心功能：Mineru支持直接读取PDF（扫描件需先OCR）、Word文档，自动提取文本、表格、图片描述（多模态数据），还能保留文档结构（章节、页码、段落关系）。
- 操作要点：
  1. 安装依赖：`pip install mineru python-dotenv`
  2. 解析代码示例：
     ```python
     from mineru import Mineru

     # 初始化解析器
     parser = Mineru()
     # 解析文档（支持批量传入多文件）
     doc_data = parser.parse("金融研报.pdf")  # 或 "公司财报.docx"
     # 提取结构化数据（文本+表格+元数据）
     texts = [item["content"] for item in doc_data["blocks"]]  # 文本内容
     metas = [{"page": item["page"], "chapter": item["chapter"]} for item in doc_data["blocks"]]  # 页码/章节等元数据
     ```
- 优势：相比传统解析工具，Mineru能更好保留金融文档的表格结构（如财务数据表格），避免文本错乱。

#### 2. 构建RAG知识库（适配金融文档特点）
- 核心目标：将解析后的文本+元数据转化为可检索的向量，搭配大模型实现精准问答。
- 关键步骤：
  1. 文本分片：金融文档多含长段落，按“章节+段落”拆分（单片段500-800字），保留表格完整性（用`pandas`辅助处理表格文本）。
  2. 向量存储：选用轻量向量库（如FAISS、Chroma），搭配金融领域适配的Embedding模型（如`bge-large-zh`、通义千问Embedding）：
     ```python
     from langchain.vectorstores import Chroma
     from langchain.embeddings import HuggingFaceEmbeddings

     # 初始化Embedding模型（金融文本适配）
     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
     # 构建向量库（文本+元数据关联）
     vector_db = Chroma.from_texts(texts, embeddings, metadatas=metas, persist_directory="./finance_rag_db")
     vector_db.persist()
     ```
  3. 检索配置：设置“相似性检索+元数据过滤”（如按“财务分析”章节筛选相关片段），提升金融问答精准度。

#### 3. RAG问答交互（支持金融场景深度查询）
- 核心逻辑：用户提问→检索知识库相关片段→拼接上下文→调用大模型生成答案（附数据来源溯源）。
- 完整代码示例（结合OpenAI/开源大模型）：
  ```python
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import RetrievalQAWithSourcesChain

  # 初始化大模型（可选开源模型如Qwen、DeepSeek）
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
  # 构建问答链（带来源追溯）
  qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vector_db.as_retriever(search_kwargs={"k": 3}),  # 取Top3相关片段
      return_source_documents=True
  )

  # 金融场景问答示例
  query = "这份研报中4Paradigm的2024年毛利率是多少？"
  result = qa_chain({"question": query})

  # 输出结果（含答案+来源，符合金融合规）
  print("答案：", result["answer"])
  print("数据来源：", [doc.metadata for doc in result["source_documents"]])
  ```

### 二、金融场景优化技巧（关键加分项）
1. 表格处理：Mineru解析的表格会转为文本格式，可先用`pandas`还原为DataFrame，再将“表格内容+表头说明”合并为文本片段，方便检索（如财务指标对比表）。
2. 来源溯源：在元数据中加入“文档名称+页码+章节”，问答结果自动标注来源，适配金融研报的合规要求。
3. 模型选择：若处理中文金融文档，优先用`bge-large-zh-finance`（金融领域微调版Embedding）和Qwen-7B-Finance（开源金融大模型），提升专业术语理解能力。
4. 批量处理：支持传入多份PDF/Word（如多季度财报、行业研报），向量库自动合并，实现跨文档问答（如“对比2023和2024年行业毛利率变化”）。

### 三、常见问题解决
- 扫描件PDF解析乱码：先用电容屏OCR工具（如天若OCR）转为可编辑PDF，再用Mineru解析。
- 检索结果不精准：调整文本分片长度（长文档拆更细）、增加检索TopK值（如k=5），或更换金融专用Embedding模型。
- 大模型生成幻觉：在Prompt中强制要求“仅基于检索到的文档内容回答，无相关信息时明确说明”。