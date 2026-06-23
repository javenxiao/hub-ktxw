# 【作业1】
## 1.数据管理的接口
### 1）上传 PDF 文档
- Endpoint  
`POST /api/data/upload_pdf`
- 功能描述：  
上传一个 PDF（图文混排），系统会自动进行：
PDF 解析（文本抽取、图像抽取）
embedding 生成
存入向量库

- Request
  
  **Header** ：  
`Content-Type: multipart/form-data`

  **Form-Data 内容** ：

<div align="center">

| 字段 | 类型 | 是否必须 | 描述 |
|:-----:|:-----:|:-----:|:-----:|
|file|File|是| PDF文档 |
|metadata|JSON(String)|否|	可选元数据，如文档类型、标签
		
</div> 

- Response

```json
{
  "status": "success",
  "document_id": "doc_202501011233",
  "pages": 87,
  "images_extracted": 35,
  "message": "PDF uploaded and indexed successfully."
}
```
### 2）删除 PDF 文档
- Endpoint  
`DELETE /api/data/delete/{document_id}`

- 功能描述：  
  删除指定文档
- request
  
  **Path 参数**  

<div align="center">

| 字段 | 类型 | 描述 |
|:-----:|:-----:|:-----:|
|document_id| string| PDF 文档唯一 ID |

</div> 

- Response

```json
{
  "status": "success",
  "document_id": "doc_202501011233",
  "message": "Document and associated embeddings deleted."
}
```

### 3）查询已上传文档列表  
- Endpoint  
`GET /api/data/list`

- 功能
     1. 列出所有已上传的 PDF 文档
     2. 返回每个文档的多模态解析状态、统计数、完备度
     3. 可以**分页**、**搜索**、**排序**
     4. 方便前端展示、后台监控、RAG 管理

- Request
  #### Query 参数（可选）

<div align="center">
  
| 参数        | 类型       | 说明                                                                   |
| --------- | :--------: | -------------------------------------------------------------------- |
| page      | int      | 页码（默认 1）                                                             |
| page_size | int      | 每页数量（默认 20）                                                          |
| keyword   | string   | 文档名模糊搜索                                                              |
| status    | enum     | 过滤文档解析状态，例如：`uploaded` / `parsed` / `embeddings_generated` / `error` |
| sort_by   | string   | 排序字段（uploaded_at / chunk_count / image_count 等）                      |
| order     | asc/desc | 排序方式                                                                 |

</div> 

- Response

```json

{
  "total": 3,
  "page": 1,
  "page_size": 20,
  "documents": [
    {
      "document_id": "doc_202501011233",
      "title": "艾力斯-公司深度报告商业化成绩显著产品矩阵持续拓宽-25070718页.pdf",
      "pages": 87,
      "file_size": 1423456,
      "uploaded_at": "2025-01-01 10:23:11",
      
      "status": "embeddings_generated",   
      "parse_progress": 100,
      "embedding_progress": 100,

      "statistics": {
        "chunk_count": 312,
        "text_chunk_count": 280,
        "image_count": 48,
        "table_count": 12,
        "multimodal_units": 60
      },

      "error_message": null
    }
  ]
}


```

### 4）获取文档的抽取内容（调试用）
- Endpoint

`GET /api/data/{document_id}/content`

- 功能描述：  
返回指定文档的全部抽取内容，内容已经 按图文 chunk 划分，每个 chunk 包含：
     1. 文本
     2. 关联图像
     3. OCR 文本（图中文字）
     4. 文本周边上下文（nearby text）
     5. chunk 元信息（id、页码、来源位置等）

- Request
  - **Path 参数**

    <div align="center">

    | 字段        | 类型   | 描述               |
    |:-----------:|:------:|--------------------|
    | document_id | string | PDF 文档唯一 ID     |

    </div>

  - **Query 参数（可选）**

    <div align="center">

    | 字段                 | 类型 | 描述                           |
    |----------------------|:----:|--------------------------------|
    | include_images       | bool | 是否返回图像信息（默认 true）   |
    | include_ocr          | bool | 是否返回 OCR 文本（默认 true）  |
    | include_nearby_text  | bool | 是否返回邻近文本（默认 true）   |

    </div>


- Response

```json
{
  "document_id": "doc_202501011233",
  "title": "艾力斯-公司深度报告商业化成绩显著产品矩阵持续拓宽-25070718页",
  "pages": 87,
  "chunks": [
    {
      "chunk_id": "blk_001",
      "page": 12,
      "text": "内容1",
      "images": [
        {
          "image_id": "img_001",
          "image_path": "data/pdf123/images/img_p12_001.png",
          "caption": "趋势图",
          "ocr_text": "colorbar 0–20"
        }
      ],
      "nearby_text": [
        {
          "unit_id": "blk_002",
          "text": "内容2"
        }
      ]
    },
    {
      "chunk_id": "blk_002",
      "page": 12,
      "text": "内容2",
      "images": [
        {
          "image_id": "img_001",
          "image_path": "data/pdf123/images/img_p12_001.png",
          "caption": "趋势图",
          "ocr_text": "colorbar 0–20"
        }
      ],
      "nearby_text": []
    }
  ]
}

```

## 2.多模态检索接口
`POST /api/retrieval/search`

- 功能描述：
     1. 输入仅文本
     2. 输入仅图像
     3. 输入同时包含文本 + 图像
     4. 跨模态检索（如用图片找对应的文本位置）

- Request
  - **Header** ：  
  `Content-Type: application/json`

  - **请求体字段（完整定义）**
```json
{
  "query_text": "请问图3中太阳能板的转换效率是多少？",
  "query_image": null,
  "embedding_type": "multimodal", 
  "top_k": 5,
  "modal_weight": {
    "text": 0.7,
    "image": 0.3
  },
  "filters": {
    "document_ids": ["doc_20250101_123456"],
    "page_range": [1, 20],
    "content_type": ["text", "image"]
  }
}

```

  - **字段解释**
      - query_text（可选）
        - 用户查询文本；
        - 可以为空（例如仅图像检索）。
      - query_image（可选）
        - Base64编码的图片；
        - 如果你有上传图片检索的需求，这个字段就很好用。
      - embedding_type（必填）
        - text	只用文本 embedding
        - image	只用图像 embedding
        - multimodal	使用 text + image 融合检索（推荐）
      - top_k（必填）
        - 返回前 K 个候选文本 / 图像块

      - modal_weight（可选）
        - 多模态融合检索的权重，例如：
          ```json
          {
              "text": 0.7,
              "image": 0.3
           }
          ```
      - filters（可选）
        
        <div align="center">
			
        | 字段           | 作用                          |
        | ------------ | --------------------------- |
        | document_ids | 限定检索的文档                     |
        | page_range   | 限定页码区间                      |
        | content_type | 返回内容类型（text / image / both） |
        
		</div>

- Response
  返回最相关的 chunk（文本段） 或 图片单元（image snippet），带有来源信息和相关性得分。

    ```json
    {
      "results": [
        {
          "chunk_id": "doc_20250101_123456_p12_c07",
          "document_id": "doc_20250101_123456",
          "page": 12,
          "content_type": "text",
          "text": "图3显示了光伏组件平均转换效率为18.7%。",
          "image_url": null,
          "score": 0.87,
          "source": {
            "document_title": "光伏发电白皮书",
            "page": 12,
            "chunk_index": 7
          }
        },
        {
          "chunk_id": "doc_20250101_123456_p12_img03",
          "document_id": "doc_20250101_123456",
          "page": 12,
          "content_type": "image",
          "text": null,
          "image_url": "/static/doc_2025/.../p12_img03.png",
          "score": 0.81,
          "source": {
            "document_title": "光伏发电白皮书",
            "page": 12,
            "image_index": 3
          }
        }
      ]
    }

    ```
  - **返回字段说明**

    <div align="center">

    | 字段           | 说明                     |
    | ------------ | ---------------------- |
    | chunk_id     | chunk 的唯一 ID           |
    | document_id  | 源 PDF 编号               |
    | page         | 页码                     |
    | content_type | text / image           |
    | text         | 文本 chunk（如果是文本）        |
    | image_url    | 图像 chunk URL（如果是图片）    |
    | score        | 相关性（归一化）               |
    | source       | 可解释性：来源页码、chunk 索引、图编号 |
		
	</div>

## 3.多模态问答接口
`POST /api/qa/ask`

- 功能描述：
     1. 接收用户的**自然语言问题（+可选图像）**
     2. 调用多模态检索模块，获取 **图文 chunk**
     3. 进行 **图文关联推理**
     4. 生成答案
     5. 返回 **可解释的来源引用**

- 不负责：
	1. 文档上传
    2. embedding 生成
    3. chunk 构建

- Request
  - **Header** ：  
  `Content-Type: application/json`
  - **请求体字段（完整定义）**

  ```json
  {
    "question": "图3中风速最大的区域在哪里？",
    "question_image": null,
    "retrieval": {
      "top_k": 6,
      "modalities": ["text", "image"],
      "document_ids": []
    },
    "answer_options": {
      "language": "zh",
      "max_tokens": 512,
      "return_sources": true
    }
  }

  ```
  
  - **字段解释**
    - question(必填)
      - 用户的自然语言问题
      - 默认是文本，但可以是 **引用图像的描述型问题**
    - question_image（可选）

      <div align="center">

      | 类型     | 说明           |
      | ------ | ------------ |
      | string | Base64 编码的图像 |
      | null   | 无图像问题        |

	  </div>

	- retrieval（检索控制参数）
      <div align="center">

	  | 字段           | 说明                |
	  | ------------ | ----------------- |
	  | top_k        | 检索 chunk 数        |
	  | modalities   | 使用的模态（text/image） |
	  | document_ids | 限定文档范围（空=全库）      |
    
      </div>

    - answer_options（生成控制）
      <div align="center">
      
      | 字段             | 说明       |
      | -------------- | -------- |
      | language       | 输出语言     |
      | max_tokens     | 最大回答长度   |
      | return_sources | 是否返回引用来源 |
      
      </div>

- Response
  - 成功响应示例
    ```json
    {
      "answer": "根据图3显示，风速最大的区域位于图像右上角的红色区域，对应于沿海高地。",
      "confidence": 0.82,
      "sources": [
        {
          "document_id": "doc_20240101",
          "document_title": "区域风场分析报告",
          "page": 5,
          "chunk_id": "chunk_005",
          "type": "image+text",
          "excerpt": "图3显示了不同区域的风速分布，红色区域代表最大风速。",
          "image_id": "img_003",
          "figure_caption": "图3 不同区域风速分布图"
        }
      ]
    }

  	```
    
  - 字段解释
    - answer
      - 最终生成的自然语言回答
      - 应该是 推理后的结果，而非直接摘抄
    - confidence（可选但推荐）
      - 模型对回答的置信度（0–1）
        可基于：
          - 检索相似度
          - 多源一致性
          - LLM self-eval
    - sources（可解释性核心）
      - 每个 source 至少包含：
        <div align="center">
			
		| 字段             | 说明                        |
		| -------------- | ------------------------- |
		| document_id    | 文档唯一 ID                   |
		| document_title | 文档标题                      |
		| page           | 页码                        |
		| chunk_id       | chunk ID                  |
		| type           | text / image / image+text |
		| excerpt        | 相关文本摘要                    |
		| image_id       | 若有关联图像                    |
		| figure_caption | 图像说明                      |
        </div>

    📌 这是评测中“可解释性”的关键加分点
