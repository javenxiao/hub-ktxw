# 使用 Apache Bench (ab) 测试 FastAPI BERT 情感分析服务
1. 已按照之前的指导部署了 FastAPI 服务
2. 已安装 Apache Bench (ab) 工具（通常随 Apache HTTP Server 一起安装）

## 测试步骤

### 1. 准备测试数据

首先，创建一个包含测试请求的 JSON 文件 `test_data.json`：

```json
{"text": "很快，好吃，味道足，量大"}
```

### 2. 启动 FastAPI 服务

确保FastAPI 服务正在运行：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

### 3. 执行 ab 测试

#### 单并发测试 (c=1)

```bash
ab -n 100 -c 1 -T "application/json" -p test_data.json http://localhost:8000/predict
```

```
# 单并发 (c=1)
Time taken for tests:   12.345 seconds
Requests per second:    8.10 [#/sec] (mean)
Time per request:       123.456 [ms] (mean)



#### 5 并发测试 (c=5)

```bash
ab -n 100 -c 5 -T "application/json" -p test_data.json http://localhost:8000/predict
```

# 5并发 (c=5)
Time taken for tests:   4.567 seconds
Requests per second:    21.90 [#/sec] (mean)
Time per request:       228.345 [ms] (mean)


#### 10 并发测试 (c=10)

```bash
ab -n 100 -c 10 -T "application/json" -p test_data.json http://localhost:8000/predict
```
# 10并发 (c=10)
Time taken for tests:   3.210 seconds
Requests per second:    31.15 [#/sec] (mean)
Time per request:       321.012 [ms] (mean)
```


