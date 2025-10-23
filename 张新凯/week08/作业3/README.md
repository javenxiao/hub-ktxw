为了运行这个应用，你需要安装 FastAPI 和 Uvicorn：

```bash
pip install "fastapi[all]" uvicorn
```

要运行这个应用，在保存代码的目录下打开终端，执行以下命令：

```bash
uvicorn main:app --reload
```