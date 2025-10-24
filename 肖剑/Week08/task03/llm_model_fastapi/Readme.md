运行步骤：
1. 创建requirements.txt文件后使用pip install -r requirements.txt 安装依赖项
2. 运行服务：
	python main.py 或者使用 uvicorn main:app --host localhost --port 8000 --reload
3. 验证服务：
	服务启动后，访问以下地址
	API 文档: http://localhost:8000/docs
	测试端点: http://localhost:8000/test
	健康检查: http://localhost:8000/health