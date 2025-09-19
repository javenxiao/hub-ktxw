## 部署运行04RAG项目

> 项目接口遵循RESTful规范

[toc]

### 1. 运行结果

#### 1.1 @app.post("/v1/embedding") 文本编码

![task02-1.1.1](.\asset\task02-1.1.1.png)

#### 1.2 @app.post("/v1/knowledge_base") 新增知识库

调用api，并查询数据库有新增记录

![task02-1.1.2](.\asset\task02-1.1.2.png)

#### 1.3 @app.post("/v1/document") 新增文档

调用api，并查询数据库有新增记录，在es中查看新增chunk数据

![task02-1.1.3](.\asset\task02-1.1.3.png)

#### 1.4 @app.get("/v1/document") 查询文档

成功

![task02-1.1.4](.\asset\task02-1.1.4.png)

#### 1.5 @app.delete("/v1/document") 删除文档报错

遇到问题，已解决；解决过程在`2.2`。

#### 1.6 @app.post("/chat") 基于RAG的聊天

成功。

![task02-1.1.5](.\asset\task02-1.1.5.png)

---

### 2. 遇到问题

#### 2.1 /docs无法加载

swagger自动加载api文档很好用。最终下载github仓库中dist静态资源到`static/swagger-ui`，并添加路由解决。

```python
app = FastAPI(
    title="RAG API",
    docs_url=None,  # 禁用默认文档
    redoc_url=None  # 禁用默认ReDoc
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


# 自定义 Swagger UI 路由
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon-32x32.png"
    )
```

#### 2.2 @app.delete("/v1/knowledge_base")报错

调用api报错5xx，顺着报错去debug

![task02-2.2.2](.\asset\task02-2.2.2.png)

![task02-2.2.1](.\asset\task02-2.2.1.png)

最后也是成功解决。