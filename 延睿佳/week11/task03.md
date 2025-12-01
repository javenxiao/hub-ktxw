## 作业3：Milvus中多模态数据结构设计

Milvus主要存储**向量化特征**及其**元数据**。针对文本、图像、表格三类数据，可设计以下数据结构。

### 表结构设计：`collection_multimodal`

| 字段名 | 类型 | 含义 |
|--------|------|------|
| `id` | String (PK) | 数据唯一标识 |
| `data_type` | Enum("text", "image", "table") | 数据类型 |
| `embedding` | FloatVector(dimension=N) | 向量表示（由不同Encoder生成） |
| `content` | VarChar / JSON | 原始内容或内容摘要 |
| `metadata` | JSON | 源信息、标签、时间戳等 |

### 嵌入维度设计

| 数据类型 | 向量来源 | 向量维度示例 |
|-----------|-----------|---------------|
| 文本 (`text`) | 文本嵌入模型（BGE / E5） | 768 |
| 图像 (`image`) | 视觉编码器（CLIP-ViT） | 512 |
| 表格 (`table`) | 表格编码模型（TAPAS / TaBERT） | 1024 |

### 示例插入数据
```python
{
  "id": "uuid-123",
  "data_type": "image",
  "embedding": [0.12, 0.98, ...],
  "content": "bridge_structure.jpg",
  "metadata": {"source": "civil_dataset", "tags": ["bridge"]}
}
```

### 检索时流程
1. 根据输入类型选择合适Encoder（如BGE、CLIP、TaBERT）。  
2. 生成向量 → 在Milvus中按相似度检索。  
3. 返回结果并传递至RAG生成模块。
