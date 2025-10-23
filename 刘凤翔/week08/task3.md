使用FastAPI部署这两个文档,以下是完整的实现过程记录：

## 1. 首先创建一个主应用文件 `main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn

from task2_01提示词 import classify_by_prompt
from task2_02tools import classify_by_tools, IntentDomainNerTask

app = FastAPI(
    title="NLP信息抽取API",
    description="基于阿里云千问模型的信息抽取服务",
    version="1.0.0"
)


class TextRequest(BaseModel):
    """文本请求模型"""
    text: str
    method: Optional[str] = "prompt"  # prompt 或 tools


class ClassificationResponse(BaseModel):
    """分类响应模型"""
    domain: str
    intent: str
    slots: Dict[str, Any]
    method: str


@app.get("/")
async def root():
    """根端点，返回API信息"""
    return {
        "message": "NLP信息抽取API服务",
        "endpoints": {
            "/health": "健康检查",
            "/classify": "文本分类和信息抽取"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "NLP Information Extraction API"}


@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: TextRequest):
    """
    文本分类和信息抽取端点
    
    Args:
        request: 包含文本和方法的请求对象
        
    Returns:
        分类结果，包含领域、意图和实体槽位
    """
    try:
        if request.method == "tools":
            # 使用tools方法
            result = classify_by_tools(request.text)
            if result is None:
                raise HTTPException(status_code=500, detail="Tools classification failed")
            
            # 转换结果为统一格式
            return ClassificationResponse(
                domain=result.domain,
                intent=result.intent,
                slots={
                    **result.entity,
                    "Src": result.Src,
                    "Des": result.Des
                },
                method="tools"
            )
        else:
            # 使用prompt方法（默认）
            result_str = classify_by_prompt(request.text)
            
            # 解析JSON结果
            try:
                import json
                result_data = json.loads(result_str)
                
                return ClassificationResponse(
                    domain=result_data.get("domain", ""),
                    intent=result_data.get("intent", ""),
                    slots=result_data.get("slots", {}),
                    method="prompt"
                )
            except json.JSONDecodeError:
                # 如果返回的不是标准JSON，尝试提取信息
                return ClassificationResponse(
                    domain="unknown",
                    intent="unknown",
                    slots={"raw_response": result_str},
                    method="prompt"
                )
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/classify/prompt")
async def classify_with_prompt(request: TextRequest):
    """专门使用prompt方法进行分类"""
    request.method = "prompt"
    return await classify_text(request)


@app.post("/classify/tools")
async def classify_with_tools(request: TextRequest):
    """专门使用tools方法进行分类"""
    request.method = "tools"
    return await classify_text(request)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发时启用热重载
        workers=1
    )
```

## 2. 创建配置文件 `config.py`

```python
import os

class Settings:
    """应用配置"""
    APP_NAME = "NLP信息抽取API"
    VERSION = "1.0.0"
    
    # API配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # 模型配置
    MODEL_NAME = "qwen-plus"
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # 环境变量检查
    @classmethod
    def validate_environment(cls):
        """验证必要的环境变量"""
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("DASHSCOPE_API_KEY environment variable is required")


settings = Settings()
```

## 3. 创建依赖文件 `requirements.txt`

```txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
openai==1.3.0
python-dotenv==1.0.0
typing-extensions==4.8.0
```

## 4. 创建环境配置文件 `.env.example`

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
HOST=0.0.0.0
PORT=8000
```

## 5. 创建启动脚本 `run.py`

```python
#!/usr/bin/env python3
"""
NLP信息抽取API启动脚本
"""

import uvicorn
from main import app
from config import settings

if __name__ == "__main__":
    # 验证环境变量
    settings.validate_environment()
    
    print(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    print(f"Server running on http://{settings.HOST}:{settings.PORT}")
    print(f"API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,  # 生产环境设为False
        access_log=True
    )
```

## 6. 创建Dockerfile（可选）`Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run.py"]
```

## 7. 创建API测试文件 `test_api.py`

```python
import requests
import json

def test_classification():
    """测试API分类功能"""
    
    base_url = "http://localhost:8000"
    
    # 测试健康检查
    response = requests.get(f"{base_url}/health")
    print("Health check:", response.json())
    
    # 测试prompt方法
    test_text = "帮我查询一下北京到天津的汽车票"
    
    data = {
        "text": test_text,
        "method": "prompt"
    }
    
    response = requests.post(f"{base_url}/classify", json=data)
    print("Prompt method result:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    # 测试tools方法
    data["method"] = "tools"
    response = requests.post(f"{base_url}/classify", json=data)
    print("\nTools method result:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_classification()
```

## 部署和使用说明

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 设置环境变量
```bash
export DASHSCOPE_API_KEY=your_actual_api_key
```

### 3. 启动服务
```bash
# 方式1: 直接运行
python run.py

# 方式2: 使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问API文档
启动后访问：http://localhost:8000/docs

### 5. API使用示例

**请求示例：**
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "播放周杰伦的七里香",
       "method": "prompt"
     }'
```

**响应示例：**
```json
{
  "domain": "music",
  "intent": "PLAY",
  "slots": {
    "artist": "周杰伦",
    "song": "七里香"
  },
  "method": "prompt"
}
```

## 主要特性

1. **双模式支持**：支持prompt和tools两种分类方法
2. **RESTful API**：标准的HTTP接口设计
3. **自动文档**：内置Swagger UI文档
4. **错误处理**：完善的异常处理机制
5. **类型安全**：使用Pydantic进行数据验证
6. **易于扩展**：模块化设计，便于添加新功能
