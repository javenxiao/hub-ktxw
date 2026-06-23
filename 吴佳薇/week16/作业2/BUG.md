BUG 1：Embedding 的形状（1D vs 2D）与 dtype（float64 vs float32）
    -问题：FAISS 要求输入给 index.add() 和 index.search() 的数组通常为 numpy.ndarray，dtype = np.float32 且为 2D (n_queries, dim)。当前代码没有强制转换 dtype/形状。示例函数 get_embedding 返回 float64（默认），可能导致 FAISS 抛错或隐式转换且性能受影响。
    -修复：在 store 与 call 中对 embedding 做标准化处理，确保：np.asarray(..., dtype=np.float32)，并且若结果是 1D（单个向量），将其 .reshape(1, -1)。
BUG 2：Redis 列表顺序与 FAISS 向量 id 顺序不一致（LPUSH 导致顺序不对）
    -问题： FAISS index.add(embedding) 会按顺序把向量追加到索引（first add -> id 0，next -> id 1，...）；原代码使用 LPUSH 将 prompt 插入 Redis 列表头（左侧），使 Redis 列表顺序与 FAISS id 顺序往往相反，从而在 call 中用 FAISS 返回的索引位置 i 去取 pormpts[i] 得到 错误的 prompt，最终 MGET 取到错误的回答或顺序错乱。
    -修复：不要直接把 prompt 字符串作为 list 的元素作为索引映射，而是存储一个稳定的 ID（例如 SHA256(prompt)）作为 list 元素，并把 name:key:ID 映射到 answer。
BUG 3：clear_cache 中删除 key 的实现错误与文件删除未检查
    -问题1：clear_cache 使用 self.redis.delete(*pormpts)，但 pormpts 是 prompt 文本（或我们修复后可能是 key_id），并且原代码没有加上 key 前缀（self.name + "key:"），因此不能正确删除回答 keys。
    -问题2：clear_cache 在删除索引文件时直接调用 os.unlink(...)，如果文件不存在会抛异常。
    -问题3：如果 pormpts 为空，self.redis.delete(*pormpts) 会在无参数情况下抛 TypeError。
    -修复1：使用 lrange 获得的 list 元素（应为 key_id）构造要删除的键名（f"{self.name}:key:{key_id}"），然后 delete(*keys)。在调用 delete 前检查 keys 非空。
    -修复2：删除索引文件前用 os.path.exists(...) 检查，或捕获 FileNotFoundError。
BUG 4：直接使用 prompt 拼接成 Redis key（安全/长度/字符问题）
    -问题：原代码使用 self.name + "key:" + q 直接拼接 prompt 做 key。如果 prompt 很长（超出 Redis key 长度限制或包含空格/二进制字符），或含有冒号冲突，会引起问题。还会泄露 prompt 内容到键名中（可能的安全/隐私问题）。
    -修复：使用哈希（如 SHA256）或 uuid 生成稳定、短小的 key id，使用这个 id 存储 value 和在列表里维护顺序（见 BUG 2 的示例）。这样也避免 key 名称过长或包含特殊字符。
BUG 5：zip(prompt, response) 的长度不一致问题
    -问题：如果 prompt 与 response 列表长度不一致，zip 将按最短长度截断，导致部分 prompt/response 被忽略且没有报错，容易造成数据不一致。
    -修复：在 store 之前显式检查长度：如果 prompt 是列表则确保 len(prompt) == len(response)，否则抛出 ValueError 或作相应处理。
