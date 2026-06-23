# Redis List 和 Set 操作指南

## Redis List（列表）操作

Redis List 是一个双向链表，支持从头部或尾部进行插入和删除操作。

### 基本操作

#### 1. 添加元素

##### LPUSH - 从左侧（头部）插入元素
```bash
LPUSH key value [value ...]
```
- **功能**：将一个或多个值插入到列表的头部（左侧）
- **返回值**：执行后列表的长度
- **示例**：
  ```bash
  LPUSH mylist "world"
  LPUSH mylist "hello"
  # 结果：["hello", "world"]
  ```

##### RPUSH - 从右侧（尾部）插入元素
```bash
RPUSH key value [value ...]
```
- **功能**：将一个或多个值插入到列表的尾部（右侧）
- **返回值**：执行后列表的长度
- **示例**：
  ```bash
  RPUSH mylist "hello"
  RPUSH mylist "world"
  # 结果：["hello", "world"]
  ```

##### LPUSHX - 仅当列表存在时从左侧插入
```bash
LPUSHX key value
```
- **功能**：只有当列表存在时，才从左侧插入元素
- **返回值**：执行后列表的长度，如果列表不存在返回 0

##### RPUSHX - 仅当列表存在时从右侧插入
```bash
RPUSHX key value
```
- **功能**：只有当列表存在时，才从右侧插入元素
- **返回值**：执行后列表的长度，如果列表不存在返回 0

#### 2. 删除元素

##### LPOP - 从左侧弹出元素
```bash
LPOP key [count]
```
- **功能**：移除并返回列表的第一个元素（左侧）
- **返回值**：被移除的元素，列表为空时返回 nil
- **示例**：
  ```bash
  LPOP mylist
  ```

##### RPOP - 从右侧弹出元素
```bash
RPOP key [count]
```
- **功能**：移除并返回列表的最后一个元素（右侧）
- **返回值**：被移除的元素，列表为空时返回 nil
- **示例**：
  ```bash
  RPOP mylist
  ```

##### BLPOP - 阻塞式从左侧弹出
```bash
BLPOP key [key ...] timeout
```
- **功能**：阻塞式地从列表左侧弹出元素，如果列表为空则等待直到有元素或超时
- **返回值**：包含键名和元素的数组，超时返回 nil
- **示例**：
  ```bash
  BLPOP mylist 10  # 等待最多 10 秒
  ```

##### BRPOP - 阻塞式从右侧弹出
```bash
BRPOP key [key ...] timeout
```
- **功能**：阻塞式地从列表右侧弹出元素，如果列表为空则等待直到有元素或超时
- **返回值**：包含键名和元素的数组，超时返回 nil

##### LREM - 移除指定元素
```bash
LREM key count value
```
- **功能**：根据 count 的值移除列表中与 value 相等的元素
- **count 参数**：
  - `count > 0`：从表头开始搜索，移除 count 个与 value 相等的元素
  - `count < 0`：从表尾开始搜索，移除 count 个与 value 相等的元素
  - `count = 0`：移除所有与 value 相等的元素
- **返回值**：被移除元素的数量
- **示例**：
  ```bash
  LREM mylist 2 "hello"  # 从头部移除 2 个 "hello"
  ```

#### 3. 查询操作

##### LRANGE - 获取列表范围内的元素
```bash
LRANGE key start stop
```
- **功能**：返回列表中指定区间内的元素
- **参数**：
  - `start`：起始位置（0 表示第一个元素）
  - `stop`：结束位置（-1 表示最后一个元素）
- **返回值**：指定区间内的元素列表
- **示例**：
  ```bash
  LRANGE mylist 0 -1  # 获取所有元素
  LRANGE mylist 0 2   # 获取前 3 个元素
  ```

##### LINDEX - 通过索引获取元素
```bash
LINDEX key index
```
- **功能**：通过索引获取列表中的元素
- **参数**：
  - `index`：索引位置（0 表示第一个，-1 表示最后一个）
- **返回值**：指定索引位置的元素，索引超出范围返回 nil
- **示例**：
  ```bash
  LINDEX mylist 0  # 获取第一个元素
  ```

##### LLEN - 获取列表长度
```bash
LLEN key
```
- **功能**：返回列表的长度
- **返回值**：列表的长度，如果列表不存在返回 0
- **示例**：
  ```bash
  LLEN mylist
  ```

#### 4. 修改操作

##### LSET - 设置指定索引的元素值
```bash
LSET key index value
```
- **功能**：通过索引设置列表元素的值
- **返回值**：操作成功返回 OK，索引超出范围返回错误
- **示例**：
  ```bash
  LSET mylist 0 "new_value"
  ```

##### LINSERT - 在指定元素前后插入
```bash
LINSERT key BEFORE|AFTER pivot value
```
- **功能**：在列表的某个元素前或后插入新元素
- **参数**：
  - `BEFORE`：在 pivot 元素之前插入
  - `AFTER`：在 pivot 元素之后插入
  - `pivot`：参考元素
  - `value`：要插入的值
- **返回值**：执行后列表的长度，如果 pivot 不存在返回 -1
- **示例**：
  ```bash
  LINSERT mylist BEFORE "world" "hello"
  ```

##### LTRIM - 修剪列表
```bash
LTRIM key start stop
```
- **功能**：只保留指定区间内的元素，不在区间内的元素将被删除
- **返回值**：操作成功返回 OK
- **示例**：
  ```bash
  LTRIM mylist 0 4  # 只保留前 5 个元素
  ```

#### 5. 高级操作

##### RPOPLPUSH - 从右侧弹出并插入到另一个列表
```bash
RPOPLPUSH source destination
```
- **功能**：从源列表的右侧弹出一个元素，并将其插入到目标列表的左侧
- **返回值**：被移动的元素
- **示例**：
  ```bash
  RPOPLPUSH mylist otherlist
  ```

##### BRPOPLPUSH - 阻塞式弹出并插入
```bash
BRPOPLPUSH source destination timeout
```
- **功能**：阻塞式地从源列表右侧弹出元素并插入到目标列表左侧
- **返回值**：被移动的元素，超时返回 nil

---

## Redis Set（集合）操作

Redis Set 是一个无序的、不重复的字符串集合。

### 基本操作

#### 1. 添加元素

##### SADD - 添加元素到集合
```bash
SADD key member [member ...]
```
- **功能**：将一个或多个成员元素加入到集合中
- **返回值**：被添加到集合中的新元素的数量（不包括已存在的元素）
- **示例**：
  ```bash
  SADD myset "hello"
  SADD myset "world" "foo"
  ```

#### 2. 删除元素

##### SREM - 移除集合中的元素
```bash
SREM key member [member ...]
```
- **功能**：移除集合中的一个或多个成员元素
- **返回值**：被成功移除的元素数量
- **示例**：
  ```bash
  SREM myset "hello"
  ```

##### SPOP - 随机移除并返回元素
```bash
SPOP key [count]
```
- **功能**：随机移除并返回集合中的一个或多个元素
- **返回值**：被移除的元素，集合为空时返回 nil
- **示例**：
  ```bash
  SPOP myset      # 移除并返回一个随机元素
  SPOP myset 3    # 移除并返回 3 个随机元素
  ```

#### 3. 查询操作

##### SMEMBERS - 获取集合所有成员
```bash
SMEMBERS key
```
- **功能**：返回集合中的所有成员
- **返回值**：集合中的所有成员
- **示例**：
  ```bash
  SMEMBERS myset
  ```

##### SCARD - 获取集合成员数量
```bash
SCARD key
```
- **功能**：返回集合中元素的数量
- **返回值**：集合中元素的数量，集合不存在返回 0
- **示例**：
  ```bash
  SCARD myset
  ```

##### SISMEMBER - 判断元素是否在集合中
```bash
SISMEMBER key member
```
- **功能**：判断 member 元素是否是集合 key 的成员
- **返回值**：如果元素是集合的成员返回 1，否则返回 0
- **示例**：
  ```bash
  SISMEMBER myset "hello"
  ```

##### SRANDMEMBER - 随机获取元素（不移除）
```bash
SRANDMEMBER key [count]
```
- **功能**：随机返回集合中的一个或多个元素，但不移除
- **参数**：
  - `count > 0`：返回 count 个不重复的元素
  - `count < 0`：返回 count 个可能重复的元素
- **返回值**：随机元素
- **示例**：
  ```bash
  SRANDMEMBER myset
  SRANDMEMBER myset 3
  ```

#### 4. 集合运算

##### SINTER - 交集
```bash
SINTER key [key ...]
```
- **功能**：返回多个集合的交集
- **返回值**：交集成员的列表
- **示例**：
  ```bash
  SINTER set1 set2 set3
  ```

##### SINTERSTORE - 交集并存储
```bash
SINTERSTORE destination key [key ...]
```
- **功能**：计算多个集合的交集并将结果存储到 destination 集合中
- **返回值**：结果集中的成员数量
- **示例**：
  ```bash
  SINTERSTORE result set1 set2
  ```

##### SUNION - 并集
```bash
SUNION key [key ...]
```
- **功能**：返回多个集合的并集
- **返回值**：并集成员的列表
- **示例**：
  ```bash
  SUNION set1 set2
  ```

##### SUNIONSTORE - 并集并存储
```bash
SUNIONSTORE destination key [key ...]
```
- **功能**：计算多个集合的并集并将结果存储到 destination 集合中
- **返回值**：结果集中的成员数量
- **示例**：
  ```bash
  SUNIONSTORE result set1 set2
  ```

##### SDIFF - 差集
```bash
SDIFF key [key ...]
```
- **功能**：返回第一个集合与其他集合的差集
- **返回值**：差集成员的列表
- **示例**：
  ```bash
  SDIFF set1 set2  # 返回在 set1 中但不在 set2 中的元素
  ```

##### SDIFFSTORE - 差集并存储
```bash
SDIFFSTORE destination key [key ...]
```
- **功能**：计算多个集合的差集并将结果存储到 destination 集合中
- **返回值**：结果集中的成员数量
- **示例**：
  ```bash
  SDIFFSTORE result set1 set2
  ```

#### 5. 移动操作

##### SMOVE - 移动元素
```bash
SMOVE source destination member
```
- **功能**：将 member 元素从 source 集合移动到 destination 集合
- **返回值**：如果元素成功移动返回 1，如果元素不在 source 集合中返回 0
- **示例**：
  ```bash
  SMOVE set1 set2 "hello"
  ```

#### 6. 扫描操作

##### SSCAN - 迭代集合中的元素
```bash
SSCAN key cursor [MATCH pattern] [COUNT count]
```
- **功能**：迭代集合中的元素
- **参数**：
  - `cursor`：游标，从 0 开始
  - `MATCH pattern`：匹配模式
  - `COUNT count`：每次迭代返回的元素数量（建议值）
- **返回值**：包含新游标和元素数组的数组
- **示例**：
  ```bash
  SSCAN myset 0 MATCH "h*" COUNT 10
  ```

---
