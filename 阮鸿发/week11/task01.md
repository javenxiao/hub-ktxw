# 多模态RAG项目接口服务

## 一、数据管理接口
### 1.1 数据上传接口
- **接口地址**：`/data/upload`
- **请求方法**：POST
- **请求类型**：`multipart/form-data`（文件上传）
- **请求参数**：
  | 参数名       | 类型       | 必选 | 说明                                  |
  |--------------|------------|------|---------------------------------------|
  | files        | File数组   | 是   | 多模态文件，支持txt、md、jpg、png、wav、mp3等 |
  | tags         | String     | 否   | 数据标签（多个用逗号分隔，如“产品、手机”） |
  | description  | String     | 否   | 数据描述（长度≤500字）                |
  | index_lib    | String     | 否   | 目标索引库名称（默认使用默认库）      |

### 1.2 数据查询接口
- **接口地址**：`/data/query`
- **请求方法**：GET
- **请求参数**（URL参数）：
  | 参数名       | 类型       | 必选 | 说明                                  |
  |--------------|------------|------|---------------------------------------|
  | data_id      | String     | 否   | 数据ID（精准查询）                    |
  | file_type    | String     | 否   | 文件类型（text/image/audio）          |
  | start_time   | String     | 否   | 上传开始时间（ISO格式，如2024-05-01T00:00:00） |
  | end_time     | String     | 否   | 上传结束时间（ISO格式）               |
  | keyword      | String     | 否   | 关键词（匹配文件名、标签、描述）      |
  | page         | Integer    | 否   | 页码（默认1）                         |
  | page_size    | Integer    | 否   | 每页数量（默认20，最大100）           |
  | index_lib    | String     | 否   | 索引库名称（默认查询所有库）          |

### 1.3 数据更新接口
- **接口地址**：`/data/update`
- **请求方法**：PUT
- **请求参数**（JSON）：
```json
{
  "data_id": "data-789",
  "new_file": null, // 可选，需更新文件时上传（multipart/form-data格式）
  "tags": "产品、手机、2024新款", // 可选，更新标签
  "description": "某品牌2024旗舰手机实拍图", // 可选，更新描述
  "index_lib": "product_new_lib" // 可选，迁移至其他索引库
}
```

### 1.4 数据删除接口
- **接口地址**：`/data/delete`
- **请求方法**：DELETE
- **请求参数**（JSON）：
```json
{
  "data_ids": ["data-789", "data-790"], // 支持批量删除
  "index_lib": "product_lib" // 可选，指定索引库（避免误删）
}
```

---

## 二、多模态检索接口
### 2.1 单一模态检索接口
- **接口地址**：`/retrieval/single`
- **请求方法**：POST
- **请求参数**（multipart/form-data，根据模态类型选择）：
  | 参数名       | 类型       | 必选 | 说明                                  |
  |--------------|------------|------|---------------------------------------|
  | query_type   | String     | 是   | 检索模态类型（text/image/audio）      |
  | text_query   | String     | 否   | 文本检索词（query_type=text时必选）   |
  | file_query   | File       | 否   | 模态文件（query_type=image/audio时必选） |
  | top_n        | Integer    | 否   | 返回数量（默认10，最大50）            |
  | threshold    | Float      | 否   | 相似度阈值（0-1，默认0.6）            |
  | index_lib    | String     | 否   | 目标索引库（默认所有库）              |

### 2.2 混合模态检索接口
- **接口地址**：`/retrieval/mixed`
- **请求方法**：POST
- **请求参数**（multipart/form-data）：
  | 参数名       | 类型       | 必选 | 说明                                  |
  |--------------|------------|------|---------------------------------------|
  | text_query   | String     | 否   | 文本检索词（至少选择一种模态）        |
  | image_query  | File       | 否   | 图像检索文件                          |
  | audio_query  | File       | 否   | 音频检索文件                          |
  | weights      | JSON String| 否   | 模态权重（如{"text":0.4,"image":0.6}，默认均分） |
  | top_n        | Integer    | 否   | 返回数量（默认10）                    |
  | threshold    | Float      | 否   | 相似度阈值（默认0.6）                 |
  | index_lib    | String     | 否   | 目标索引库                            |

---

## 三、多模态问答接口
### 3.1 单轮问答接口
- **接口地址**：`/qa/single`
- **请求方法**：POST
- **请求参数**（multipart/form-data）：
  | 参数名       | 类型       | 必选 | 说明                                  |
  |--------------|------------|------|---------------------------------------|
  | question     | String     | 是   | 用户问题（如“这款手机的拍照功能怎么样？”） |
  | image_input  | File       | 否   | 图像输入（辅助问答）                  |
  | audio_input  | File       | 否   | 音频输入（辅助问答）                  |
  | top_n        | Integer    | 否   | 检索关联数据数量（默认5）             |
  | llm_version  | String     | 否   | 大模型版本（默认v1，支持v2/pro）      |
  | enable_trace | Boolean    | 否   | 是否开启溯源（默认true）              |
  | index_lib    | String     | 否   | 检索索引库（默认所有库）              |

### 3.2 多轮问答接口
- **接口地址**：`/qa/multi`
- **请求方法**：POST
- **请求参数**（multipart/form-data）：
```json
{
  "session_id": "session-789", // 会话ID（首次请求可不传，自动生成）
  "question": "它的电池容量是多少？", // 当前问题
  "image_input": null, // 可选，当前轮图像输入
  "audio_input": null, // 可选，当前轮音频输入
  "top_n": 5,
  "llm_version": "v1",
  "enable_trace": true
}
```
