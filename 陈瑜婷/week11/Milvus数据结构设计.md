# Milvus 数据结构设计文档

## 多模态RAG项目 - 向量数据库Schema设计

---

## 设计目标

在多模态RAG系统中，需要存储和检索三种主要数据类型：
- **文本块（Text Chunks）**：文档分段后的文本内容
- **图片（Images）**：文档中提取的图片、图表、示意图等
- **表格（Tables）**：文档中的结构化表格数据


### Collection名称: `multimodal_rag_knowledge`

| 字段名 | 字段类型 | 维度/长度 | 是否主键 | 索引类型 | 描述 |
|--------|---------|----------|---------|---------|------|
| **primary_key** | INT64 | - | ✅ Yes | - | Milvus自动生成的主键ID |
| **content_type** | VARCHAR | 50 | ❌ | - | 内容类型：text/image/table |
| **document_id** | INT64 | - | ❌ | - | 所属文档ID（关联MySQL） |
| **chunk_id** | INT64 | - | ❌ | - | 块ID（关联MySQL） |
| **page_number** | INT32 | - | ❌ | - | 所在页码 |
| **content** | VARCHAR | 5000 | ❌ | - | 文本内容/表格markdown/图片描述 |
| **image_path** | VARCHAR | 500 | ❌ | - | 图片本地路径（仅image类型） |
| **table_data** | VARCHAR | 10000 | ❌ | - | 表格JSON数据（仅table类型） |
| **metadata** | JSON | - | ❌ | - | 扩展元数据字段 |
| **text_bge_vector** | FLOAT_VECTOR | 512 | ❌ | IVF_FLAT | BGE文本编码向量 |
| **text_clip_vector** | FLOAT_VECTOR | 512 | ❌ | IVF_FLAT | CLIP文本编码向量 |
| **image_clip_vector** | FLOAT_VECTOR | 512 | ❌ | IVF_FLAT | CLIP图片编码向量 |
| **created_at** | INT64 | - | ❌ | - | 创建时间戳 |

---

## 字段详细说明

### 1. 核心标识字段

#### `primary_key` (INT64)
- Milvus自动生成的唯一主键
- 用于向量数据库的内部索引
- 需要在MySQL中记录此ID，用于关联

#### `content_type` (VARCHAR 50)
- 枚举值：
  - `text`: 普通文本块
  - `image`: 图片（照片、图表、示意图）
  - `table`: 表格
- 用于检索时的类型过滤

#### `document_id` (INT64)
- 关联MySQL中的document表
- 用于按文档过滤检索结果
- 支持"只在某些文档中检索"的需求

#### `chunk_id` (INT64)
- 在MySQL中的唯一chunk记录ID
- 用于回溯到原始数据
- 文本/图片/表格在MySQL中都有对应记录

---

### 2. 位置信息字段

#### `page_number` (INT32)
- 内容所在的PDF页码
- 用于答案溯源："该信息来自第X页"

---

### 3. 内容字段

#### `content` (VARCHAR 5000)
根据不同类型存储不同内容：
- **text类型**：分段后的文本内容（用于显示给用户）
- **image类型**：图片的描述/标题（如"图1: RAG系统架构"）
- **table类型**：表格的Markdown格式文本（用于LLM理解）

#### `image_path` (VARCHAR 500)
- 仅对image类型有效
- 存储图片的本地文件路径
- 示例：`./images/doc123_page5_img1.png`

#### `table_data` (VARCHAR 10000)
- 仅对table类型有效
- 存储表格的结构化数据（JSON格式）
- 示例：
```json
{
    "headers": ["产品", "销量", "增长率"],
    "rows": [
        ["产品A", "1000", "15%"],
        ["产品B", "800", "10%"]
    ]
}
```

#### `metadata` (JSON)
- 扩展元数据字段，灵活存储其他信息
- 示例：
```json
{
    "caption": "图1: RAG系统架构",
    "table_title": "2024年销售数据",
    "bbox": [100, 200, 500, 600],  // 图片/表格在原页面的坐标
    "ocr_confidence": 0.95,
    "language": "zh-CN"
}
```

---

### 4. 向量字段

#### `text_bge_vector` (FLOAT_VECTOR 512)
- **用途**：纯文本语义检索（单模态）
- **编码对象**：
  - text类型：chunk文本内容
  - table类型：表格的markdown文本
  - image类型：不编码（为NULL或零向量）
- **模型**：BGE-small-zh-v1.5

#### `text_clip_vector` (FLOAT_VECTOR 512)
- **用途**：跨模态检索（文本→图片）
- **编码对象**：
  - text类型：chunk文本内容
  - table类型：表格的markdown文本
  - image类型：图片描述文本
- **模型**：Chinese-CLIP（文本编码器）

#### `image_clip_vector` (FLOAT_VECTOR 512)
- **用途**：跨模态检索（图片→文本，图片→图片）
- **编码对象**：
  - image类型：图片的视觉特征
  - table类型：表格渲染为图片后编码（可选）
  - text类型：不编码（为NULL或零向量）
- **模型**：Chinese-CLIP（图像编码器）

#### `created_at` (INT64)
- Unix时间戳
- 用于数据管理和调试
