# 训练模型

```commandline
python3 train/train_bert.py
```

# 压测服务

```commandline
cd test/
ab -n 100 -c 10 -p test_data.json -T 'application/json' -H 'accept: application/json' 'http://127.0.0.1:8000/v1/bert/classify'
```

# 接口

```commandline
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/bert/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "request_id": "123456",
  "request_text": "很棒，很快，很好吃"
}'
```

# 部署

```commandline
fastapi run main.py
```