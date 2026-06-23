# SemanticCache.py Bug 分析文档


## Bug 1: `call` 方法中的索引映射错误

**位置**: 第70-74行

**问题描述**:
```70:74:llm_cache/SemanticCache.py
filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]

pormpts = self.redis.lrange(self.name + "list", 0, -1)
print("pormpts", pormpts)
filtered_prompts = [pormpts[i] for i in filtered_ind]
```

**问题分析**:
1. `filtered_ind` 使用的是 `enumerate(dis[0])` 的索引 `i`，这个 `i` 是 `dis[0]` 数组的位置索引（0-99），而不是 faiss 索引的实际值
2. faiss 的 `search` 方法返回 `(dis, ind)`，其中 `ind[0]` 才包含实际的 faiss 索引值
3. 应该使用 `ind[0][i]` 来获取对应的 faiss 索引

**修复**:
```python
# 应该修改为：
filtered_ind = [ind[0][i] for i, d in enumerate(dis[0]) if d < self.distance_threshold]
```

---

## Bug 2: `clear_cache` 方法中的删除逻辑错误

**位置**: 第80-81行

**问题描述**:
```80:81:llm_cache/SemanticCache.py
pormpts = self.redis.lrange(self.name + "list", 0, -1)
self.redis.delete(*pormpts)
```

**问题分析**:
1. `pormpts` 包含的是 prompt 字符串本身，而不是 Redis key
2. `delete` 方法需要的是 key 名称（如 `self.name + "key:" + prompt`），而不是 prompt 内容
3. 这导致无法正确删除 Redis 中的缓存数据

**修复**:
```python
prompts = self.redis.lrange(self.name + "list", 0, -1)
# 删除所有对应的 key-value 对
keys_to_delete = [self.name + "key:" + q.decode() for q in prompts]
if keys_to_delete:
    self.redis.delete(*keys_to_delete)
self.redis.delete(self.name + "list")
```

---
