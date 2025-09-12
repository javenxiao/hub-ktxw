### RAG类

#### chat_with_rag

如果是第一轮对话，先进行信息检索，然后将检索到的内容组合到提示词模版中，调用模型进行推理。

内容检索方式有两种，分别是全文检索以及向量检索。如果没有开启重排序，会组合两种检索方式的排名进行排序，如果开启重排序，则使用模型对召回内容进行重排序（这里的重排序代码好像没起作用）。

排名的组合方式：

$$
\frac{1}{(60 + rank_{fulltext})} + \frac{1}{(60 + rank_{vector})}
$$

#### extract_content

读取pdf的内容，然后按页以及按chunk将内容和对应的向量放入到es中。chunk的切分方式为固定长度切分，向量使用配置的向量模型生成，数据格式如下：

按页存放：

```python
{
  "document_id": document_id,
  "knowledge_id": knowledge_id,
  "page_number": page_number,
  "chunk_id": 0, # 先存储每一也所有内容
  "chunk_content": current_page_text,
  "chunk_images": [],
  "chunk_tables": [],
  "embedding_vector": embedding_vector
}
```

按chunk存放：

```python
{
  "document_id": document_id,
  "knowledge_id": knowledge_id,
  "page_number": page_number,
  "chunk_id": chunk_idx,
  "chunk_content": page_chunks[chunk_idx - 1],
  "chunk_images": [],
  "chunk_tables": [],
  "embedding_vector": embedding_vector[chunk_idx - 1]
}
```

#### get_rank

对文本进行打分，比较两个文本之间的相似度