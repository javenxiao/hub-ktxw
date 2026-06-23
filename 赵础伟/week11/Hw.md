# Task 1
## /data
## POST/PUT
## Request:
```
{
  "data_type": "string", // 数据类型，例如 "text", "image", "video"
  "data_content": "string", // 数据内容，文本内容或文件的Base64编码
  "metadata": {
    "title": "string", // 数据标题
    "description": "string", // 数据描述
    "tags": ["string"], // 标签列表
    "source": "string" // 数据来源
  }
}
```
## Response
```
{
  "status": "success", // 上传状态，"success" 或 "error"
  "data_id": "string", // 数据唯一标识符
  "message": "string" // 错误信息或成功提示
}
```

# Task 2

## 文本提问：
  1. 使用BGE模型提取向量
  2. 在向量库中对text_bge_vector进行检索
  3. 使用CLIP Text提取向量
  4. 在向量库中对text_image_vector进行检索
  5. 对检索到的结果重排序
  6. 将重排序后的结果和用户提问交给qwen模型
  7. 返回qwen模型的回答
## 文本 + 图提问：
  1. 使用BGE模型提取文本向量
  2. 在向量库中对text_bge_vector进行检索
  3. 使用CLIP Image提取图片向量
  4. 在向量库中对image_vector进行检索
  5. 对检索到的结果重排序
  6. 将重排序后的结果和用户提问交给qwen模型
  7. 返回qwen模型的回答

# Task 3

| 字段名	| 数据类型	| 描述| 
|-------------------| -------- |-------|
| id	| String	| 数据的唯一标识符，用于快速索引和检索。| 
| modality	| String	| 数据的模态类型，例如 "text"、"image"、"table"。| 
| vector	| Float[]	| 数据的向量表示，用于向量检索。| 
| metadata	| JSON Object	数据的元信息，例如标题、描述、标签、来源等。| 
| raw_data	| String (可选)	| 原始数据的存储路径或内容（如文本内容、图像URL、表格文件路径等）。| 