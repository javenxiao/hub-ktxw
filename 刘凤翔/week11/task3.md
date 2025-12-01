
## 🏗️ 整体数据结构设计

### 1. 基础元数据字段
```python
# 基础元数据字段设计
base_schema = {
    "id": "INT64",  # 主键
    "data_type": "VARCHAR",  # 数据类型: text/image/table
    "source_id": "VARCHAR",  # 原始数据源ID
    "created_time": "VARCHAR",  # 创建时间
    "updated_time": "VARCHAR",  # 更新时间
    "metadata": "JSON",  # 扩展元数据
}
```

### 2. 文本数据专用字段
```python
text_schema = {
    "text_content": "VARCHAR(65535)",  # 文本内容
    "text_length": "INT32",  # 文本长度
    "language": "VARCHAR",  # 语言类型
    "category": "VARCHAR",  # 文本分类
}
```

### 3. 图像数据专用字段
```python
image_schema = {
    "image_path": "VARCHAR",  # 图片路径/URL
    "image_format": "VARCHAR",  # 图片格式
    "image_size": "INT64",  # 文件大小
    "image_dimensions": "VARCHAR",  # 图片尺寸 "1920x1080"
    "dominant_colors": "JSON",  # 主色调
}
```

### 4. 表格数据专用字段
```python
table_schema = {
    "table_name": "VARCHAR",  # 表名
    "table_schema": "JSON",  # 表结构
    "row_count": "INT32",  # 行数
    "column_count": "INT32",  # 列数
    "table_summary": "VARCHAR",  # 表格摘要
}
```

## 🔄 向量字段设计策略

### 方案一：统一向量字段（推荐）
```python
# 统一的向量字段设计
vector_schema = {
    # 多模态统一向量 (使用多模态模型如CLIP、BLIP等)
    "multimodal_vector": "FLOAT_VECTOR(512)",
    
    # 文本专用向量 (使用文本优化模型如BGE、Sentence-BERT)
    "text_semantic_vector": "FLOAT_VECTOR(768)", 
    
    # 图像专用向量 (使用视觉模型如ResNet、ViT)
    "image_visual_vector": "FLOAT_VECTOR(512)",
    
    # 表格结构化向量 (使用表格编码模型)
    "table_structural_vector": "FLOAT_VECTOR(256)",
}
```

### 方案二：按模态分集合（大规模场景）
```python
# 文本专用集合
text_collection_schema = {
    "text_bge_vector": "FLOAT_VECTOR(768)",
    "text_clip_vector": "FLOAT_VECTOR(512)",
}

# 图像专用集合  
image_collection_schema = {
    "image_clip_vector": "FLOAT_VECTOR(512)",
    "image_resnet_vector": "FLOAT_VECTOR(2048)",
}

# 表格专用集合
table_collection_schema = {
    "table_semantic_vector": "FLOAT_VECTOR(512)",
    "table_schema_vector": "FLOAT_VECTOR(256)",
}
```

## 📊 完整集合架构

### 主集合设计（推荐用于中小规模）
```python
def create_multimodal_collection():
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    
    # 基础字段
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="data_type", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="created_time", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="updated_time", datatype=DataType.VARCHAR, max_length=50)
    
    # 内容字段
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="file_format", datatype=DataType.VARCHAR, max_length=20)
    
    # 元数据字段
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    
    # 向量字段
    schema.add_field(field_name="multimodal_vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="text_semantic_vector", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="image_visual_vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="table_structural_vector", datatype=DataType.FLOAT_VECTOR, dim=256)
    
    # 索引配置
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE", 
        "params": {"nlist": 1024}
    }
    
    collection = client.create_collection(
        collection_name="multimodal_rag",
        schema=schema,
        index_params=index_params
    )
    
    return collection
```

## 🎯 数据插入示例

### 文本数据插入
```python
def insert_text_data(text_content, metadata=None):
    # 文本特征提取
    text_semantic_vector = get_text_bge_features([text_content])[0]
    multimodal_vector = get_clip_text_features([text_content])[0]
    
    data = {
        "data_type": "text",
        "content": text_content,
        "text_length": len(text_content),
        "language": detect_language(text_content),
        "text_semantic_vector": text_semantic_vector.tolist(),
        "multimodal_vector": multimodal_vector.tolist(),
        "source_id": f"text_{hash(text_content)}",
        "created_time": get_current_time(),
        "metadata": metadata or {}
    }
    
    return client.insert("multimodal_rag", [data])
```

### 图像数据插入
```python
def insert_image_data(image_path, description=None):
    image = Image.open(image_path)
    
    # 多模态特征提取
    image_visual_vector = get_clip_image_features([image])[0]
    multimodal_vector = image_visual_vector  # CLIP图像向量可直接用作多模态向量
    
    # 如果有关联文本，也提取文本向量
    if description:
        text_vector = get_clip_text_features([description])[0]
    else:
        text_vector = [0] * 512
    
    data = {
        "data_type": "image",
        "file_path": image_path,
        "file_format": image_path.split('.')[-1],
        "image_dimensions": f"{image.width}x{image.height}",
        "image_visual_vector": image_visual_vector.tolist(),
        "multimodal_vector": multimodal_vector.tolist(),
        "text_semantic_vector": text_vector.tolist(),  # 可选：图像描述向量
        "source_id": f"image_{hash(image_path)}",
        "created_time": get_current_time(),
        "metadata": {
            "description": description,
            "file_size": os.path.getsize(image_path)
        }
    }
    
    return client.insert("multimodal_rag", [data])
```

### 表格数据插入
```python
def insert_table_data(table_path, table_name, summary=None):
    # 读取表格数据
    if table_path.endswith('.csv'):
        df = pd.read_csv(table_path)
    else:
        df = pd.read_excel(table_path)
    
    # 表格语义向量（基于表格内容摘要）
    table_content = generate_table_summary(df, summary)
    table_semantic_vector = get_text_bge_features([table_content])[0]
    multimodal_vector = get_clip_text_features([table_content])[0]
    
    # 表格结构向量（基于表结构和统计信息）
    table_structural_vector = encode_table_structure(df)
    
    data = {
        "data_type": "table",
        "content": table_content,  # 表格文本摘要
        "file_path": table_path,
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "table_structural_vector": table_structural_vector.tolist(),
        "text_semantic_vector": table_semantic_vector.tolist(),
        "multimodal_vector": multimodal_vector.tolist(),
        "source_id": f"table_{hash(table_path)}",
        "created_time": get_current_time(),
        "metadata": {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "summary_stats": generate_summary_stats(df)
        }
    }
    
    return client.insert("multimodal_rag", [data])
```

## 🔍 多模态检索策略

### 统一检索接口
```python
def multimodal_search(query, data_type=None, top_k=10):
    """
    支持多种查询类型的统一检索
    """
    if isinstance(query, str):
        # 文本查询
        query_vector = get_clip_text_features([query])[0]
        anns_field = "multimodal_vector"
        
    elif isinstance(query, Image.Image):
        # 图像查询
        query_vector = get_clip_image_features([query])[0]
        anns_field = "multimodal_vector"
        
    elif isinstance(query, pd.DataFrame):
        # 表格查询
        table_content = generate_table_summary(query)
        query_vector = get_text_bge_features([table_content])[0]
        anns_field = "text_semantic_vector"
    
    # 构建过滤条件
    filter_expr = None
    if data_type:
        filter_expr = f"data_type == '{data_type}'"
    
    # 执行检索
    results = client.search(
        collection_name="multimodal_rag",
        data=[query_vector.tolist()],
        anns_field=anns_field,
        filter=filter_expr,
        limit=top_k,
        output_fields=["id", "data_type", "content", "file_path", "metadata"]
    )
    
    return process_search_results(results)
```

### 跨模态检索示例
```python
# 文本搜图
text_to_image_results = multimodal_search(
    query="黑色的笔记本电脑", 
    data_type="image",
    top_k=5
)

# 图搜文
image_to_text_results = multimodal_search(
    query=user_uploaded_image,
    data_type="text", 
    top_k=5
)

# 表格相似搜索
table_similarity_results = multimodal_search(
    query=reference_dataframe,
    data_type="table",
    top_k=3
)
```

## ⚡ 性能优化设计

### 1. 索引策略
```python
# 为不同向量字段创建专用索引
index_configs = {
    "multimodal_vector": {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024}
    },
    "text_semantic_vector": {
        "index_type": "HNSW", 
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    },
    "image_visual_vector": {
        "index_type": "IVF_SQ8",
        "metric_type": "COSINE", 
        "params": {"nlist": 512}
    }
}
```

### 2. 分区策略（大规模数据）
```python
# 按数据类型分区
partition_names = ["text_partition", "image_partition", "table_partition"]

for partition in partition_names:
    client.create_partition(
        collection_name="multimodal_rag",
        partition_name=partition
    )
```

## 📈 扩展性考虑

### 动态字段支持
```python
# 利用Milvus的动态字段存储扩展信息
dynamic_metadata = {
    "text_quality_score": 0.95,
    "image_quality_score": 0.88,
    "table_data_quality": 0.92,
    "extracted_keywords": ["笔记本电脑", "游戏本", "RTX显卡"],
    "content_categories": ["电子产品", "电脑"],
    "access_frequency": 156,
    "last_accessed": "2024-01-15 10:30:00"
}
```
