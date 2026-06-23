## 阅读 SemanticCache.py，找出其中存在bug
> 阅读下 https://github.com/redis/redis-vl-python 的项目和已有的代码

### ttl值固定容易引发缓存雪崩
指的是，如果大量缓存的生存时间设置为相同的固定值，它们会在同一时刻过期，从而大量请求冲向后端服务，
导致数据库压力骤增、响应变慢、甚至崩溃，进而引发整个系统的连锁故障

可设置动态TTL值，例如给固定值加一个0-10%的随机偏移量；缓存预热，主动加载|刷新热点数据

### 多级键名构造不清晰
self.name + "list"这种拼接方式生成的键名（如mynamelist）可读性较差；使用明确的分隔符（如冒号:），这是redis中常用的写法

### call()的改进
#### 检索向量中的K个数，应添加到call()入参
#### 返回值个数可以约定返回数据
#### 解码添加异常处理

### clear_cache()存在多个问题
#### 在lrange和delete之间，列表内容可能已被修改
使用Redis事务（如pipeline）或Lua脚本保证原子性

#### 删除list中元素错误，且没有删除key
```commandline
pormpts = self.redis.lrange(self.name + "list", 0, -1)
self.redis.delete(*pormpts)
```

#### 缺乏错误处理
使用try-except，并添加一致性补偿处理

### faiss的IO操作缺乏异常处理
对于`faiss.write_index(self.index, f"{self.name}.index")`和`faiss.read_index(f"{self.name}.index")`等IO操作，应添加异常处理