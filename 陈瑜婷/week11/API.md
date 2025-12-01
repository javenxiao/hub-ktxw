# 多模态RAG问答系统 - API接口文档

## 项目概述

本文档定义了多模态RAG问答系统的所有API接口，包括数据管理、多模态检索和多模态问答三大功能模块。

## 技术栈

- **服务框架**: FastAPI
- **数据存储**: SQLite/MySQL + Milvus
- **多模态模型**: Qwen-VL、CLIP、BGE
- **文档解析**: DeepSeek-OCR / MinerU

---

## 一、数据管理接口

### 1.1 健康检查

**接口地址**: `GET /health`

**功能描述**: 检查后端服务状态，包括数据库连接和向量数据库连接状态。

**请求参数**: 无

**响应格式**:

```json
{
    "status": "healthy",
    "service": "Multimodal RAG Chatbot Service",
    "uptime": "123.45 seconds",
    "database": {
        "sqlite_connected": true,
        "milvus_connected": true,
        "milvus_collections": ["documents_text", "documents_image"]
    }
}
```

---

### 1.2 上传文档

**接口地址**: `POST /document/upload`

**功能描述**: 上传PDF文档到指定知识库，系统将自动解析文档，提取文本和图片，并生成向量索引。

**请求参数**:

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| file | File | 是 | PDF文档文件 |
| knowledge_base_id | int | 是 | 知识库ID（Form字段） |
| title | string | 否 | 文档标题，不填则使用文件名（Form字段） |
| description | string | 否 | 文档描述（Form字段） |


**响应格式**:

```json
{
    "status": 200,
    "message": "文档上传成功，正在后台解析",
    "data": {
        "document_id": 123,
        "filename": "example.pdf",
        "title": "RAG技术文档",
        "knowledge_base_id": 1,
        "created_at": "2025-11-12 10:30:00",
        "parse_status": "pending"
    }
}
```

**状态说明**:
- `pending`: 等待解析
- `parsing`: 解析中
- `completed`: 解析完成
- `failed`: 解析失败

---

### 1.3 获取文档列表

**接口地址**: `GET /document/list`

**功能描述**: 获取指定知识库中的所有文档列表，支持分页和排序。

**请求参数**:

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
|--------|------|------|--------|------|
| knowledge_base_id | int | 是 | - | 知识库ID |
| page_index | int | 否 | 1 | 页码，从1开始 |
| page_size | int | 否 | 20 | 每页数量 |
| order_by | string | 否 | created_at | 排序字段：created_at/updated_at/title |
| order | string | 否 | desc | 排序方式：asc/desc |
| parse_status | string | 否 | all | 过滤解析状态：all/pending/parsing/completed/failed |


**响应格式**:

```json
{
    "status": 200,
    "message": "查询成功",
    "data": {
        "documents": [
            {
                "id": 123,
                "title": "RAG技术文档",
                "filename": "example.pdf",
                "description": "关于RAG的详细介绍",
                "knowledge_base_id": 1,
                "parse_status": "completed",
                "page_count": 15,
                "chunk_count": 45,
                "image_count": 8,
                "file_size": 2048576,
                "created_at": "2025-11-12 10:30:00",
                "updated_at": "2025-11-12 10:35:00"
            }
        ],
        "total": 100,
        "page_index": 1,
        "page_size": 10,
        "total_pages": 10
    }
}
```

---

### 1.4 获取文档详情

**接口地址**: `GET /document/{document_id}`

**功能描述**: 获取指定文档的详细信息，包括解析后的文本块和图片信息。

**请求参数**:

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| document_id | int | 是 | 文档ID（路径参数） |
| include_chunks | bool | 否 | 是否包含文本块详情，默认false |
| include_images | bool | 否 | 是否包含图片详情，默认false |


**响应格式**:

```json
{
    "status": 200,
    "message": "查询成功",
    "data": {
        "id": 123,
        "title": "RAG技术文档",
        "filename": "example.pdf",
        "description": "关于RAG的详细介绍",
        "knowledge_base_id": 1,
        "parse_status": "completed",
        "page_count": 15,
        "chunk_count": 45,
        "image_count": 8,
        "file_size": 2048576,
        "file_path": "./documents/123_example.pdf",
        "created_at": "2025-11-12 10:30:00",
        "updated_at": "2025-11-12 10:35:00",
        "chunks": [
            {
                "chunk_id": 1,
                "content": "这是第一个文本块的内容...",
                "page_number": 1,
                "milvus_primary_key": "456789"
            }
        ],
        "images": [
            {
                "image_id": 1,
                "image_path": "./images/123_1.png",
                "page_number": 3,
                "caption": "图1: RAG架构图",
                "milvus_primary_key": "456790"
            }
        ]
    }
}
```

---

### 1.5 删除文档

**接口地址**: `DELETE /document/{document_id}`

**功能描述**: 删除指定文档，包括关系型数据库记录、向量数据库中的向量、本地文件。

**请求参数**:

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| document_id | int | 是 | 文档ID（路径参数） |

**响应格式**:

```json
{
    "status": 200,
    "message": "文档删除成功",
    "data": {
        "document_id": 123,
        "deleted_chunks": 45,
        "deleted_images": 8
    }
}
```

---

### 1.6 获取知识库列表

**接口地址**: `GET /knowledge_base/list`

**功能描述**: 获取所有知识库列表。

**请求参数**: 无

**响应格式**:

```json
{
    "status": 200,
    "message": "查询成功",
    "data": {
        "knowledge_bases": [
            {
                "id": 1,
                "name": "技术文档库",
                "description": "存储技术相关文档",
                "document_count": 50,
                "created_at": "2025-11-01 10:00:00"
            },
            {
                "id": 2,
                "name": "产品文档库",
                "description": "存储产品相关文档",
                "document_count": 30,
                "created_at": "2025-11-05 15:00:00"
            }
        ],
        "total": 2
    }
}
```

---

### 1.7 创建知识库

**接口地址**: `POST /knowledge_base`

**功能描述**: 创建新的知识库。

**请求参数**:

```json
{
    "name": "新知识库",
    "description": "知识库描述"
}
```

**响应格式**:

```json
{
    "status": 200,
    "message": "知识库创建成功",
    "data": {
        "id": 3,
        "name": "新知识库",
        "description": "知识库描述",
        "created_at": "2025-11-12 11:00:00"
    }
}
```

---

## 二、多模态检索接口

### 2.1 多模态检索

**接口地址**: `POST /search`

**功能描述**: 支持四种检索模式：文本检索文本、文本检索图片、图片检索文本、图片检索图片。

**请求参数**:

```json
{
    "knowledge_base_id": 1,
    "search_type": "text2text",
    "query_text": "什么是RAG技术？",
    "query_image": null,
    "top_k": 5,
    "document_ids": [123, 124],
    "filter_type": "all"
}
```

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| knowledge_base_id | int | 是 | 知识库ID |
| search_type | string | 是 | 检索类型：text2text/text2image/image2text/image2image |
| query_text | string | 条件 | 查询文本（text2text和text2image时必填） |
| query_image | string | 条件 | 查询图片（base64编码，image2text和image2image时必填） |
| top_k | int | 否 | 返回结果数量，默认5 |
| document_ids | array | 否 | 限定检索的文档ID列表，为空则检索所有文档 |
| filter_type | string | 否 | 结果类型过滤：all/text_only/image_only，默认all |

**请求示例**:

**1. 文本检索文本 (text2text)**

```json
{
    "knowledge_base_id": 1,
    "search_type": "text2text",
    "query_text": "什么是检索增强生成？",
    "top_k": 5,
    "document_ids": [123, 124]
}
```

**2. 文本检索图片 (text2image)**

```json
{
    "knowledge_base_id": 1,
    "search_type": "text2image",
    "query_text": "RAG系统架构图",
    "top_k": 3
}
```

**3. 图片检索文本 (image2text)**

```json
{
    "knowledge_base_id": 1,
    "search_type": "image2text",
    "query_image": "base64_encoded_image_string...",
    "top_k": 5
}
```

**4. 图片检索图片 (image2image)**

```json
{
    "knowledge_base_id": 1,
    "search_type": "image2image",
    "query_image": "base64_encoded_image_string...",
    "top_k": 5
}
```

**响应格式**:

**文本检索文本响应**:

```json
{
    "status": 200,
    "message": "检索成功",
    "data": {
        "search_type": "text2text",
        "results": [
            {
                "type": "text",
                "content": "检索增强生成（RAG）是一种结合检索和生成的技术...",
                "document_id": 123,
                "document_title": "RAG技术文档",
                "chunk_id": 15,
                "page_number": 3,
                "similarity_score": 0.95,
                "milvus_primary_key": "456789"
            },
            {
                "type": "text",
                "content": "RAG系统通常包含三个主要组件...",
                "document_id": 124,
                "document_title": "AI技术综述",
                "chunk_id": 28,
                "page_number": 7,
                "similarity_score": 0.88,
                "milvus_primary_key": "456790"
            }
        ],
        "total": 2,
        "query_embedding_model": "bge-small-zh-v1.5"
    }
}
```

**文本检索图片响应**:

```json
{
    "status": 200,
    "message": "检索成功",
    "data": {
        "search_type": "text2image",
        "results": [
            {
                "type": "image",
                "image_id": 5,
                "image_path": "./images/123_5.png",
                "image_url": "/api/image/123_5.png",
                "caption": "图1: RAG系统架构",
                "document_id": 123,
                "document_title": "RAG技术文档",
                "page_number": 5,
                "similarity_score": 0.92,
                "milvus_primary_key": "456791"
            }
        ],
        "total": 1,
        "query_embedding_model": "chinese-clip-vit-base-patch16"
    }
}
```

**图片检索文本响应**:

```json
{
    "status": 200,
    "message": "检索成功",
    "data": {
        "search_type": "image2text",
        "results": [
            {
                "type": "text",
                "content": "这张图展示了RAG系统的完整架构...",
                "document_id": 123,
                "document_title": "RAG技术文档",
                "chunk_id": 20,
                "page_number": 5,
                "similarity_score": 0.89,
                "milvus_primary_key": "456792"
            }
        ],
        "total": 1,
        "query_embedding_model": "chinese-clip-vit-base-patch16"
    }
}
```

**图片检索图片响应**:

```json
{
    "status": 200,
    "message": "检索成功",
    "data": {
        "search_type": "image2image",
        "results": [
            {
                "type": "image",
                "image_id": 7,
                "image_path": "./images/124_3.png",
                "image_url": "/api/image/124_3.png",
                "caption": "图2: 类似的系统架构",
                "document_id": 124,
                "document_title": "AI技术综述",
                "page_number": 9,
                "similarity_score": 0.87,
                "milvus_primary_key": "456793"
            }
        ],
        "total": 1,
        "query_embedding_model": "chinese-clip-vit-base-patch16"
    }
}
```

---

## 三、多模态问答接口

### 3.1 多模态问答

**接口地址**: `POST /chat`

**功能描述**: 基于知识库进行多模态问答，支持文本和图片混合输入，返回答案及相关来源信息。

**请求参数**:

```json
{
    "knowledge_base_id": 1,
    "question": "根据文档，RAG技术的核心优势是什么？",
    "question_image": null,
    "document_ids": [123, 124],
    "top_k": 5,
    "enable_rerank": true,
    "temperature": 0.7,
    "max_tokens": 1000
}
```

| 参数名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| knowledge_base_id | int | 是 | 知识库ID |
| question | string | 是 | 用户问题 |
| question_image | string | 否 | 问题相关的图片（base64编码） |
| document_ids | array | 否 | 限定检索的文档ID列表，为空则检索所有文档 |
| top_k | int | 否 | 检索结果数量，默认5 |
| enable_rerank | bool | 否 | 是否启用重排序，默认true |
| temperature | float | 否 | 生成温度，默认0.7 |
| max_tokens | int | 否 | 最大生成token数，默认1000 |

**请求示例**:

**1. 纯文本问答**

```json
{
    "knowledge_base_id": 1,
    "question": "RAG技术的核心组件有哪些？",
    "top_k": 5
}
```

**2. 图文混合问答**

```json
{
    "knowledge_base_id": 1,
    "question": "这张图中的架构是什么？请详细解释。",
    "question_image": "base64_encoded_image_string...",
    "top_k": 5
}
```

**响应格式**:

```json
{
    "status": 200,
    "message": "问答成功",
    "data": {
        "answer": "根据检索到的文档内容，RAG技术的核心组件主要包括：\n\n1. **检索器（Retriever）**：负责从知识库中检索相关文档片段。\n2. **生成器（Generator）**：基于检索到的内容生成最终答案。\n3. **向量数据库**：存储文档的向量表示，支持高效检索。\n\n这些组件协同工作，使得RAG能够结合外部知识进行准确回答。",
        "sources": [
            {
                "type": "text",
                "content": "RAG系统主要由检索器和生成器两部分组成...",
                "document_id": 123,
                "document_title": "RAG技术文档",
                "page_number": 3,
                "chunk_id": 15,
                "similarity_score": 0.95
            },
            {
                "type": "image",
                "image_id": 5,
                "image_url": "/api/image/123_5.png",
                "caption": "图1: RAG系统架构",
                "document_id": 123,
                "document_title": "RAG技术文档",
                "page_number": 5,
                "similarity_score": 0.88
            }
        ],
        "model_used": "Qwen-VL",
        "retrieval_time_ms": 150,
        "generation_time_ms": 800,
        "total_time_ms": 950
    }
}
```

**字段说明**:
- `answer`: 生成的答案文本
- `sources`: 答案来源列表，包括相关文本片段和图片
- `model_used`: 使用的模型名称
- `retrieval_time_ms`: 检索耗时（毫秒）
- `generation_time_ms`: 生成耗时（毫秒）
- `total_time_ms`: 总耗时（毫秒）

---
