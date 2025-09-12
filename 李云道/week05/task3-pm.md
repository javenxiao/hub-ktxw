# 政企问答项目--RAG代码部分
[TOC]

## 1. 项目目的

> AI+政务的应用

基于各地市政府已有的政府文件、FAQ问答等数据，利用LLM技术完成智能问答系统的升级迭代（需要支持多轮问答）

## 2. RAG是什么

### 2.1 RAG原理

 RAG，全称为 **Retrieval-Augmented Generation**（检索增强生成）。它的核心思想非常直观：

**在让大语言模型（LLM）回答问题之前，先从一个外部知识库中检索相关的信息，然后把这些信息作为上下文和问题一起交给LLM，让LLM基于这些权威信息来生成答案**。

- **没有RAG的LLM**：像是闭卷考试。模型只能依靠它训练时记忆的知识（参数）来回答问题。如果问题超出了它的记忆范围，它就***可能“编造”答案***（幻觉）。

- **有RAG的LLM**：像是开卷考试。允许模型在回答前先“查阅”一堆指定的参考资料（你的知识库），然后基于这些资料组织语言来回答问题。答案更准确、更有依据。

### 2.2 RAG流程

RAG的工作流程通常分为三个核心步骤：

**Ⅰ. 检索（Retrieval）**

- **输入**：用户提出一个问题（Query）。
- **处理**：系统将用户的问题进行编码，转化为一个向量（Vector Embedding）。这个向量可以理解为问题在数学空间中的“含义”。
- **查找**：系统在一个预先构建好的“向量数据库”中，查找与这个问题向量最相似的知识片段（Chunks）。这些知识片段来自你提供的私有文档、数据库、网页等任何外部知识源。
- **输出**：系统找到最相关的几个知识片段（例如，Top 3）。

**Ⅱ. 增强（Augmentation）**

- 将**用户的原问题**和**检索到的相关知识点**组合在一起，拼装成一个新的、内容更丰富的“提示（Prompt）”。
- 例如，Prompt模板可能是：“请基于以下背景信息：`[这里插入检索到的知识]`，来回答问题：`[用户的原问题]`”。

**Ⅲ. 生成（Generation）**

- 将这个增强后的Prompt发送给大语言模型（如GPT-4、LLaMA等）。
- LLM不再依赖于其内部可能过时或不完整的知识，而是**严格遵守你提供的背景信息**来生成最终答案。
- **输出**：一个准确、有据可依的回答，并且通常模型还会注明答案的来源。

## 3. RAG代码实现

***首要条件***

将外部知识转为数据库。本项目中使用**sqlite**存储元数据、使用**es**存储chunk块转成的词向量并加速查找

***Ⅰ. 检索（R）***

- ***输入***：用户提出一个问题（query），`main.py`中`@app.post("/chat")def chat(req: RAGRequest) -> RAGResponse:`为服务入口

- ***处理与查找***：

  调用`rag_api.RAG`，输出为message: str。封装成RAGResponse类返回json格式

  ```python
  from rag_api import RAG
  message = RAG().chat_with_rag(req.knowledge_id, req.message)
  return RAGResponse(...)
  ```
  “查询”转化为向量，经过全文检索和语义检索，多路召回
  ```python
  def chat_with_rag(
      self,
      knowledge_id: int, # 知识库 哪一个知识库提问
      messages: List[Dict],
  ):
      self.query_document(query, knowledge_id)
      # 全文检索，
  	--> word_search_response = es.search(...)
      # 语义检索
      -->embedding_vector = self.get_embedding(query) # 编码
      -->vector_search_response = es.search(...)
      # RRF融合：结合两种检索结果进行相关性融合排序
      -->
  ```

  重排序，未使用，未实现

  ```python
  # 
  ```

***Ⅱ. 增强（A）***

- 拼装成一个新的、内容更丰富的“提示（Prompt）”

  ```python
  BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
  如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
  如果问题可以从资料中获得，则请逐步回答。
  
  资料：
  {#RELATED_DOCUMENT#}
  
  
  问题：{#QUESTION#}
  '''
  rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
  .replace("{#QUESTION#}", query) \
  .replace("{#RELATED_DOCUMENT#}", related_document)
  ```

***Ⅲ. 生成（G）***

- 将这个增强后的Prompt发送给大语言模型，

  ```python
  self.client = OpenAI(
      api_key=config["rag"]["llm_api_key"],
      base_url=config["rag"]["llm_base"]
  )
  
  def chat(self, messages: List[Dict], top_p: float, temperature: float) -> Any:
      completion = self.client.chat.completions.create(
          model=self.llm_model,
          messages=messages,
          top_p=top_p,
          temperature=temperature
      )
      return completion.choices[0].message
  
  rag_response = self.chat(
      [{"role": "user", "content": rag_query}],
      0.7, 0.9
  ).content
  messages.append({"role": "system", "content": rag_response})
  ```

  

- LLM严格遵守**提供的背景信息**来生成最终答案,输出：一个准确、有据可依的回答，并且通常模型还会注明答案的来源