# ChatBI 项目接口文档

## 项目概述
ChatBI 是一个基于 AI 大模型的智能对话系统，集成了股票数据分析、新闻资讯、用户管理等功能。系统采用 FastAPI 后端和 Streamlit 前端架构。

## 接口总览

### 1. 用户管理接口 (User Management)

#### 1.1 用户登录
- **接口路径**: `POST /v1/users/login`
- **请求体**:
```json
{
  "user_name": "string",
  "password": "string"
}
```
- **响应**:
```json
{
  "code": 200,
  "message": "string",
  "data": []
}
```

#### 1.2 用户注册
- **接口路径**: `POST /v1/users/register`
- **请求体**:
```json
{
  "user_name": "string",
  "password": "string",
  "user_role": "string"
}
```
- **响应**: 同上

#### 1.3 获取用户信息
- **接口路径**: `POST /v1/users/info`
- **查询参数**: `user_name`
- **响应**:
```json
{
  "code": 200,
  "message": "string",
  "data": {
    "user_id": 1,
    "user_name": "string",
    "user_role": "string",
    "register_time": "datetime",
    "status": true
  }
}
```

#### 1.4 重置密码
- **接口路径**: `POST /v1/users/reset-password`
- **请求体**:
```json
{
  "user_name": "string",
  "password": "string",
  "new_password": "string"
}
```

#### 1.5 修改用户信息
- **接口路径**: `POST /v1/users/reset-info`
- **请求体**:
```json
{
  "user_name": "string",
  "user_role": "string",
  "status": true
}
```

#### 1.6 删除用户
- **接口路径**: `POST /v1/users/delete`
- **查询参数**: `user_name`

#### 1.7 获取用户列表
- **接口路径**: `POST /v1/users/list`
- **响应**: 返回所有用户列表

### 2. 聊天对话接口 (Chat)

#### 2.1 发起对话
- **接口路径**: `POST /v1/chat/`
- **请求体**:
```json
{
  "content": "string",
  "user_name": "string",
  "session_id": "string",
  "task": "string",
  "tools": ["string"]
}
```
- **响应类型**: `StreamingResponse` (SSE 流式响应)

#### 2.2 初始化对话会话
- **接口路径**: `POST /v1/chat/init`
- **响应**:
```json
{
  "code": 200,
  "message": "ok",
  "data": {
    "session_id": "string"
  }
}
```

#### 2.3 获取对话记录
- **接口路径**: `POST /v1/chat/get`
- **查询参数**: `session_id`
- **响应**: 返回指定会话的所有消息记录

#### 2.4 删除对话会话
- **接口路径**: `POST /v1/chat/delete`
- **查询参数**: `session_id`

#### 2.5 获取用户对话列表
- **接口路径**: `POST /v1/chat/list`
- **查询参数**: `user_name`
- **响应**: 返回用户的所有对话会话

#### 2.6 对话反馈
- **接口路径**: `POST /v1/chat/feedback`
- **查询参数**: `session_id`, `message_id`, `feedback`

### 3. 股票数据接口 (Stock)

#### 3.1 获取用户收藏股票
- **接口路径**: `POST /v1/stock/list_fav_stock`
- **查询参数**: `user_name`
- **响应**: 返回用户收藏的股票列表

#### 3.2 添加收藏股票
- **接口路径**: `POST /v1/stock/add_fav_stock`
- **查询参数**: `user_name`, `stock_code`

#### 3.3 删除收藏股票
- **接口路径**: `POST /v1/stock/del_fav_stock`
- **查询参数**: `user_name`, `stock_code`

#### 3.4 清空收藏股票
- **接口路径**: `POST /v1/stock/clear_fav_stock`
- **查询参数**: `user_name`

### 4. 数据管理接口 (Data)

#### 4.1 下载数据
- **接口路径**: `POST /v1/data/download`

#### 4.2 创建数据
- **接口路径**: `POST /v1/data/create`

#### 4.3 上传数据
- **接口路径**: `POST /v1/data/upload`

#### 4.4 删除数据
- **接口路径**: `POST /v1/data/delete`
