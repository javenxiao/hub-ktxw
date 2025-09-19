# Elasticsearch 快速入门指南（从关系型数据库视角）
>作为熟悉关系型数据库(Oracle, MySQL)的开发者，学习 Elasticsearch 的关键是理解两者的核心差异

[toc]

将用 SQL 对比的方式帮你快速掌握 ES 的核心概念和 CRUD 操作。

## 核心概念对比

| 关系型数据库 (MySQL) | Elasticsearch (ES) | 说明 |
|----------------------|---------------------|------|
| 数据库 (Database)    | 索引 (Index)        | ES 的索引类似数据库 |
| 表 (Table)           | 类型 (Type)         | ES 7.x 后类型被废弃，现在直接使用索引 |
| 行 (Row)             | 文档 (Document)     | ES 的基本数据单元 |
| 列 (Column)          | 字段 (Field)        | 数据属性 |
| 主键 (Primary Key)   | _id 字段            | ES 自动生成或自定义 |
| SQL                  | Query DSL           | ES 的查询语言 |
| 索引 (Index)         | 倒排索引            | ES 的核心数据结构 |

## ES 的 CRUD 操作（使用 curl 示例）

### 1. 创建索引（类似创建数据库）
```bash
# 创建 products 索引
curl -X PUT "localhost:9200/products?pretty"
```

### 2. 插入文档（类似 INSERT）
```bash
# 插入一条产品文档（自动生成ID）
curl -X POST "localhost:9200/products/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "name": "Laptop",
  "price": 999.99,
  "description": "High-performance laptop with 16GB RAM",
  "category": "Electronics",
  "in_stock": true
}
'

# 插入文档并指定ID（类似指定主键）
curl -X PUT "localhost:9200/products/_doc/1001?pretty" -H 'Content-Type: application/json' -d'
{
  "name": "Smartphone",
  "price": 699.99,
  "description": "Latest smartphone with 5G",
  "category": "Electronics",
  "in_stock": true
}
'
```

**使用mapper定义文档**

```bash
# 创建 products 索引并定义映射
curl -X PUT "localhost:9200/products?pretty" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          }
        }
      },
      "price": {
        "type": "float"
      },
      "description": {
        "type": "text",
        "analyzer": "english"
      },
      "category": {
        "type": "keyword"
      },
      "in_stock": {
        "type": "boolean"
      },
      "created_at": {
        "type": "date",
        "format": "yyyy-MM-dd HH:mm:ss||epoch_millis"
      },
      "tags": {
        "type": "keyword"
      }
    }
  }
}
'
```



### 3. 查询文档（类似 SELECT）

#### 获取单个文档（类似 SELECT * WHERE id=）
```bash
curl -X GET "localhost:9200/products/_doc/1001?pretty"
```

#### 简单搜索（类似 WHERE 条件查询）
```bash
curl -X GET "localhost:9200/products/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match": {
      "category": "Electronics"
    }
  }
}
'

# 等价 SQL: SELECT * FROM products WHERE category = 'Electronics'
```

#### 复杂搜索（多条件查询）
```bash
curl -X GET "localhost:9200/products/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        { "match": { "category": "Electronics" } },
        { "range": { "price": { "gte": 500 } } }
      ],
      "must_not": [
        { "term": { "in_stock": false } }
      ]
    }
  }
}
'

# 等价 SQL: 
# SELECT * FROM products 
# WHERE category = 'Electronics' 
# AND price >= 500 
# AND in_stock = true
```

### 4. 更新文档（类似 UPDATE）

#### 全量更新（替换整个文档）
```bash
curl -X PUT "localhost:9200/products/_doc/1001?pretty" -H 'Content-Type: application/json' -d'
{
  "name": "Smartphone Pro",
  "price": 799.99,
  "description": "Premium smartphone with advanced camera",
  "category": "Electronics",
  "in_stock": true
}
'

# 注意：PUT 会替换整个文档，未包含的字段会被删除
```

#### 部分更新（只修改指定字段）
```bash
curl -X POST "localhost:9200/products/_update/1001?pretty" -H 'Content-Type: application/json' -d'
{
  "doc": {
    "price": 749.99,
    "description": "Premium smartphone with advanced camera and 5G"
  }
}
'

# 等价 SQL: 
# UPDATE products 
# SET price = 749.99, description = 'Premium smartphone...' 
# WHERE id = 1001
```

### 5. 删除文档（类似 DELETE）
```bash
# 删除单个文档
curl -X DELETE "localhost:9200/products/_doc/1001?pretty"

# 删除整个索引（类似 DROP DATABASE）
curl -X DELETE "localhost:9200/products?pretty"
```

### 6.查看生成的映射(mapping)

```bash
curl -X GET "localhost:9200/products/_mapping?pretty"
```

## 高级查询示例

### 全文搜索（ES 核心优势）
```bash
curl -X GET "localhost:9200/products/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match": {
      "description": "high performance camera"
    }
  }
}'

# 会匹配包含这些词的文档，按相关性排序
```

### 聚合分析（类似 GROUP BY）
```bash
curl -X GET "localhost:9200/products/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "aggs": {
    "avg_price": {
      "avg": { "field": "price" }
    },
    "by_category": {
      "terms": { "field": "category.keyword" }
    }
  }
}'

# 等价 SQL:
# SELECT category, AVG(price) 
# FROM products 
# GROUP BY category
```

本地调用图片

![ddd](.\asset\task01-es.png)

## 重要注意事项

1. **无模式 vs 强模式**：
   - ES 是 schemaless（写入时自动创建字段）
   - 生产环境建议定义映射（mapping）控制字段类型

2. **近实时性**：
   - 文档插入后约1秒才可搜索（可调整）

3. **分布式特性**：
   - 数据自动分片（shard）和复制（replica）

4. **分词与倒排索引**：
   - 文本字段会被分词（如 "high-performance" → ["high", "performance"]）
   - 这是 ES 快速全文搜索的基础

## AI给的学习建议

1. 先掌握基本 CRUD 操作
2. 理解 Query DSL 结构（query, filter, aggs）
3. 练习常用查询类型：match, term, range, bool
4. 学习聚合分析（metrics, bucket）
5. 最后研究映射和分词器

ES 的查询能力远超传统 SQL，特别适合全文搜索、复杂过滤和实时分析场景。开始时可能会觉得 Query DSL 复杂，但熟悉后会发现它比 SQL 更灵活强大。
