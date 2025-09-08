# 训练模型

```commandline
python training_code/train_bert.py
```

# 压测服务

```commandline
cd test/

ab -n 100 -c 1 -l -p data.json -T 'application/json' -H 'accept: application/json' 'http://127.0.0.1:8000/v1/text-cls/bert'
ab -n 100 -c 5 -l -p data.json -T 'application/json' -H 'accept: application/json' 'http://127.0.0.1:8000/v1/text-cls/bert'
ab -n 100 -c 10 -l -p data.json -T 'application/json' -H 'accept: application/json' 'http://127.0.0.1:8000/v1/text-cls/bert'
```

# 接口

```commandline
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/text-cls/bert' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "request_id": "string",
  "request_text": "分量少，味道也不太好"
}'
```

# 部署

```commandline
fastapi run main.py
```