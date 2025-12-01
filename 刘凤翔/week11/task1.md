## 🏗️ 接口架构设计

### 1. 数据管理接口

```python
# data_models.py 扩展
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# 基础响应模型
class BasicResponse(BaseModel):
    status: int
    message: str
    data: Optional[Union[dict, list]] = None

# 商品管理模型
class ProductCreate(BaseModel):
    title: str = Field(..., description="商品标题")
    image_base64: Optional[str] = Field(None, description="商品图片base64编码")

class ProductUpdate(BaseModel):
    title: Optional[str] = Field(None, description="新商品标题")
    image_base64: Optional[str] = Field(None, description="新商品图片base64编码")

class ProductResponse(BaseModel):
    id: int
    title: str
    image_path: str
    created_at: datetime
    updated_at: datetime
    milvus_primary_key: int

class ProductListResponse(BaseModel):
    products: List[ProductResponse]
    total: int
    page: int
    page_size: int
```

### 2. 多模态检索接口

```python
# 检索相关模型
class SearchRequest(BaseModel):
    search_type: str = Field(
        default="text2text",
        description="检索类型: text2text, text2image, image2text, image2image"
    )
    query_text: Optional[str] = Field(None, description="查询文本")
    query_image: Optional[str] = Field(None, description="查询图片base64")
    top_k: int = Field(default=10, ge=1, le=50, description="返回结果数量")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")

class SearchResult(BaseModel):
    id: int
    title: str
    image_path: str
    distance: float = Field(..., description="相似度距离")
    milvus_primary_key: int
    created_at: datetime
    updated_at: datetime

class SearchResponse(BaseModel):
    results: List[SearchResult]
    search_type: str
    query_info: Dict[str, Any]
    total_hits: int
```

### 3. 多模态问答接口

```python
# 问答相关模型
class QARequest(BaseModel):
    question: str = Field(..., description="用户问题")
    context_images: Optional[List[str]] = Field(None, description="上下文图片base64列表")
    chat_history: Optional[List[Dict]] = Field(None, description="对话历史")
    search_top_k: int = Field(default=5, description="检索相关文档数量")
    temperature: float = Field(default=0.7, ge=0, le=1, description="生成温度")

class DocumentChunk(BaseModel):
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]

class QAResponse(BaseModel):
    answer: str
    supporting_documents: List[DocumentChunk]
    retrieved_images: List[str] = Field(..., description="检索到的相关图片路径")
    reasoning_chain: Optional[List[str]] = Field(None, description="推理链条")
    confidence: float = Field(..., ge=0, le=1, description="回答置信度")
```

## 🔌 完整接口定义

### 数据管理接口
```python
# main.py 中的接口定义

@app.get("/health")
async def health_check() -> BasicResponse:
    """服务健康检查"""
    pass

@app.post("/products", response_model=BasicResponse)
async def create_product(product: ProductCreate) -> BasicResponse:
    """创建新商品"""
    pass

@app.get("/products", response_model=BasicResponse)
async def list_products(
    page: int = 1,
    page_size: int = 20,
    order_by: str = "created_at"
) -> BasicResponse:
    """获取商品列表（分页）"""
    pass

@app.get("/products/{product_id}", response_model=BasicResponse)
async def get_product(product_id: int) -> BasicResponse:
    """获取单个商品详情"""
    pass

@app.patch("/products/{product_id}", response_model=BasicResponse)
async def update_product(
    product_id: int, 
    update_data: ProductUpdate
) -> BasicResponse:
    """更新商品信息"""
    pass

@app.delete("/products/{product_id}", response_model=BasicResponse)
async def delete_product(product_id: int) -> BasicResponse:
    """删除商品"""
    pass

@app.post("/products/batch", response_model=BasicResponse)
async def batch_create_products(products: List[ProductCreate]) -> BasicResponse:
    """批量创建商品"""
    pass
```

### 多模态检索接口
```python
@app.post("/search", response_model=BasicResponse)
async def semantic_search(search_request: SearchRequest) -> BasicResponse:
    """
    多模态语义检索
    - text2text: 文本搜文本
    - text2image: 文本搜图片  
    - image2text: 图片搜文本
    - image2image: 图片搜图片
    """
    pass

@app.post("/search/hybrid", response_model=BasicResponse)
async def hybrid_search(
    query_text: Optional[str] = None,
    query_image: Optional[str] = None,
    top_k: int = 10,
    text_weight: float = 0.5
) -> BasicResponse:
    """混合检索：同时使用文本和图片进行检索"""
    pass

@app.get("/search/similar/{product_id}", response_model=BasicResponse)
async def find_similar_products(
    product_id: int,
    top_k: int = 10,
    search_type: str = "image2image"
) -> BasicResponse:
    """根据商品ID查找相似商品"""
    pass
```

### 多模态问答接口
```python
@app.post("/qa", response_model=BasicResponse)
async def multimodal_qa(qa_request: QARequest) -> BasicResponse:
    """
    多模态问答
    - 基于用户问题检索相关商品信息
    - 结合文本和图片信息生成答案
    """
    pass

@app.post("/qa/stream")
async def stream_multimodal_qa(qa_request: QARequest):
    """流式多模态问答（用于实时显示生成过程）"""
    pass

@app.post("/qa/visual", response_model=BasicResponse)
async def visual_qa(
    question: str,
    image_base64: str,
    top_k: int = 5
) -> BasicResponse:
    """视觉问答：针对特定图片进行问答"""
    pass

@app.get("/qa/history/{session_id}", response_model=BasicResponse)
async def get_qa_history(session_id: str) -> BasicResponse:
    """获取问答会话历史"""
    pass
```

## 🎯 接口调用示例

### 1. 创建商品
```python
import base64
import requests

# 读取图片并编码
with open("product.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode()

create_data = {
    "title": "高品质笔记本电脑",
    "image_base64": image_base64
}

response = requests.post(
    "http://localhost:8000/products",
    json=create_data
)
```

### 2. 多模态检索
```python
# 文本搜图
search_data = {
    "search_type": "text2image",
    "query_text": "寻找黑色笔记本电脑",
    "top_k": 10
}

response = requests.post(
    "http://localhost:8000/search",
    json=search_data
)
```

### 3. 多模态问答
```python
qa_data = {
    "question": "推荐几款适合程序员的笔记本电脑，并说明理由",
    "search_top_k": 8,
    "temperature": 0.7
}

response = requests.post(
    "http://localhost:8000/qa", 
    json=qa_data
)
```

## 📊 错误处理设计

```python
class ErrorResponse(BaseModel):
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None

# 标准错误码
ERROR_CODES = {
    "INVALID_SEARCH_TYPE": "无效的检索类型",
    "MISSING_QUERY": "缺少查询参数", 
    "PRODUCT_NOT_FOUND": "商品不存在",
    "IMAGE_PROCESS_FAILED": "图片处理失败",
    "VECTOR_DB_ERROR": "向量数据库错误",
    "MODEL_INFERENCE_ERROR": "模型推理错误"
}
```
