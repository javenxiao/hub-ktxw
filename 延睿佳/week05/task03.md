# RAG 实现流程说明

RAG（检索增强生成，Retrieval-Augmented Generation）是一种结合 **信息检索（Retrieval）** 与 **生成模型（Generation）** 的自然语言处理方法。它的核心思想是：  
在生成模型输出答案前，从外部知识库中检索相关信息，并将其作为上下文提供给生成模型，从而提高回答的准确性和知识覆盖范围。

---

## 1. 配置与初始化

- 使用 `config.yaml` 配置运行参数，包括：
  - **设备信息**（CPU / GPU）
  - **Embedding 模型配置**
  - **Rerank 模型配置**
  - **LLM（大语言模型）配置**
  - **Chunk 切分参数**

- 读取配置后，加载所需模型：
  - Embedding 模型：`SentenceTransformer`
  - Rerank 模型：`AutoModelForSequenceClassification` + `AutoTokenizer`
  - LLM 客户端：`OpenAI` API

---

## 2. 文档解析与向量化

### 2.1 PDF 文档解析
通过 `pdfplumber` 对 PDF 文件逐页解析，提取文本内容：
1. 每页完整内容存储并向量化。
2. 将页面内容按 **chunk_size + overlap** 切分成多个片段（chunk）。
3. 对每个 chunk 生成 **向量表示**（embedding）。
4. 将结果存储至 **Elasticsearch (ES)**：
   - **chunk_info** 索引：存储每个 chunk 的文本内容与向量。
   - **document_meta** 索引：存储文档的元信息（标题、路径、摘要等）。

### 2.2 Word 文档解析
当前未实现，预留接口 `_extract_word_content`。

---

## 3. 文本编码与重排序

### 3.1 Embedding 生成
- 使用指定的 **Embedding 模型** 将文本（query 或文档片段）编码为向量。
- 示例模型：`bge-small-zh-v1.5`, `bge-base-zh-v1.5`。

### 3.2 Rerank 机制
- 使用 **Rerank 模型** 对 query 与候选文档对进行语义匹配打分。
- 示例模型：`bge-reranker-base`。
- 输出匹配分数，用于结果优化排序。

---

## 4. 检索流程

### 4.1 初步检索
针对用户 query，采用两种检索方式：
1. **全文检索**（BM25）
   - 在 ES 的 `chunk_info` 索引中进行匹配检索。
2. **语义检索**（向量检索）
   - 先对 query 编码，再在向量索引中进行最近邻检索。

### 4.2 结果融合（RRF）
- 使用 **RRF（Reciprocal Rank Fusion）** 融合 BM25 与向量检索结果。
- 融合公式：  
  \[
  score(d) = \sum \frac{1}{rank(d) + k}
  \]
  ，其中 \(k\) 是平滑因子（如 60）。

### 4.3 候选集筛选
- 取融合后的 Top-N 结果（由 `chunk_candidate` 控制）。
- 若开启 rerank，则对候选结果与 query 进行二次排序。

---

## 5. RAG 问答流程

### 5.1 构造 Prompt
- 使用模板 **BASIC_QA_TEMPLATE**：
    ```
    现在的时间是：{#TIME#}。
    你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
    如果问题无法从资料中获得，请回答无法回答。

    资料：{#RELATED_DOCUMENT#}

    问题：{#QUESTION#}
    ```
    - 将相关文档内容填充至 `{#RELATED_DOCUMENT#}`，用户问题填充至 `{#QUESTION#}`。

### 5.2 调用 LLM
- 将构造后的 Prompt 作为 `messages`，调用 OpenAI LLM 接口生成回答。

### 5.3 消息管理
- 将生成的回答追加到 `messages` 中，形成对话上下文。

---

## 6. 流程总结

1. **模型加载** ：加载 Embedding 模型与 Rerank 模型。
2. **文档入库** ：文档解析 → 分页/切分 → Embedding → 存储至 ES。
3. **用户提问** ：输入 query。
4. **检索阶段** ：BM25 + 向量检索 → RRF 融合 → 候选结果 → （可选 rerank）。
5. **生成阶段** ：构造 Prompt → 调用 LLM → 输出答案。

