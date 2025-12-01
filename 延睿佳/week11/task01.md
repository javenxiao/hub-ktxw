## 作业1：多模态RAG系统接口设计

多模态RAG（Retrieval-Augmented Generation）系统通常包括三个主要功能模块：

- **数据管理接口**：用于数据上传、更新、删除与索引。
- **多模态检索接口**：用于基于多模态输入（文本、图像、表格等）进行相似度检索。
- **多模态问答接口**：结合检索结果与生成模型，实现问答功能。

### 1. 数据管理接口

#### 接口名称：`/api/data/upload`

**功能描述：**  
上传多模态数据（文本、图像、表格）并建立索引。

**请求方式：** `POST`

**请求参数：**
```json
{
  "data_type": "text | image | table",
  "content": "（当 data_type=text 时为文本内容）",
  "file_url": "（当 data_type=image/table 时为文件URL或Base64编码）",
  "metadata": {
    "source": "dataset_name",
    "tags": ["finance", "map"],
    "id": "optional_unique_id"
  }
}
```

**返回结果：**
```json
{
  "status": "success",
  "data_id": "uuid-xxxxxx",
  "message": "Data uploaded and indexed successfully"
}
```

---

#### 接口名称：`/api/data/delete`

**功能描述：**  
根据ID删除数据。

**请求方式：** `DELETE`

**请求参数：**
```json
{
  "data_id": "uuid-xxxxxx"
}
```

**返回结果：**
```json
{
  "status": "success",
  "message": "Data deleted from Milvus"
}
```

---

### 2. 多模态检索接口

#### 接口名称：`/api/retrieve`

**功能描述：**  
根据文本、图像或表格查询多模态数据库，返回最相关的内容。

**请求方式：** `POST`

**请求参数：**
```json
{
  "query": {
    "text": "Describe the structure of this bridge",
    "image": "base64-encoded-image(optional)",
    "table": "table JSON(optional)"
  },
  "top_k": 5
}
```

**返回结果：**
```json
{
  "status": "success",
  "results": [
    {
      "data_id": "uuid-1",
      "data_type": "image",
      "similarity": 0.92,
      "content_preview": "bridge_structure.jpg",
      "metadata": {
        "source": "civil_dataset"
      }
    },
    {
      "data_id": "uuid-2",
      "data_type": "text",
      "similarity": 0.88,
      "content_preview": "A suspension bridge consists of..."
    }
  ]
}
```

---

### 3. 多模态问答接口

#### 接口名称：`/api/qa`

**功能描述：**  
用户输入文本或图像问题，系统检索相关多模态知识并生成回答。

**请求方式：** `POST`

**请求参数：**
```json
{
  "question": "What is shown in the image?",
  "image": "base64-encoded-image(optional)",
  "context_top_k": 5
}
```

**返回结果：**
```json
{
  "status": "success",
  "answer": "The image shows a suspension bridge over a river.",
  "retrieved_context": [
    {
      "data_id": "uuid-1",
      "content_preview": "bridge_structure.jpg"
    }
  ],
  "model_info": {
    "generator": "Qwen-VL-Chat",
    "retriever": "CLIP + BGE-M3"
  }
}
```
