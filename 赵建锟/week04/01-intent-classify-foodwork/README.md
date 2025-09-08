# 训练模型

```commandline
python3 training_code/train_tfidf.py
python3 training_code/train_bert.py
```

# 压测服务

```commandline
cd test/

ab -n 100 -c 100 -p data.json -T 'application/json' -H 'accept: application/json' 'http://0.0.0.0:8000/v1/text-cls/regex'
ab -n 100 -c 100 -p data.json -T 'application/json' -H 'accept: application/json' 'http://0.0.0.0:8000/v1/text-cls/tfidf'
ab -n 100 -c 100 -p data.json -T 'application/json' -H 'accept: application/json' 'http://0.0.0.0:8000/v1/text-cls/bert'
```

# 接口

```commandline
curl -X 'POST' \
  'http://0.0.0.0:8000/v1/text-cls/tfidf' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "request_id": "string",
  "request_text": "帮我播放周杰伦的歌曲"
}'
```

# 部署

```commandline
fastapi run main.py
```