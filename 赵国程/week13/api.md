### 概览

- 基础地址: `http://localhost:8000`

### 统一响应结构

- 类型: `BasicResponse`（`models/data_models.py:12`）
- 字段:
  - `code`(int): 业务状态码，`200` 表示成功
  - `message`(str): 文本信息
  - `data`(list|any): 业务数据

示例:
```
{
  "code": 200,
  "message": "ok",
  "data": []
}
```

### 健康检查

1. #### 健康检查
- Path: `/v1/healthy`（`main_server.py:21`）
- Method: GET
- RequestParam: 无
- ResponseBody:
```
{}
```

### 用户

1. #### 用户登录
- Path: `/v1/users/login`
- Method: POST
- RequestBody: `RequestForUserLogin
  - `user_name`(str)
  - `password`(str)
- ResponseBody: `BasicResponse`

2. #### 用户注册
- Path: `/v1/users/register`
- Method: POST
- RequestBody: `RequestForUserRegister
  - `user_name`(str)
  - `password`(str)
  - `user_role`(str)
- ResponseBody: `BasicResponse`

3. #### 用户重置密码
- Path: `/v1/users/reset-password`
- Method: POST
- RequestBody: `RequestForUserResetPassword`
  - `user_name`(str)
  - `password`(str)
  - `new_password`(str)
- ResponseBody: `BasicResponse`

4. #### 获取用户信息
- Path: `/v1/users/info`
- Method: POST
- RequestParam:
  - `user_name`(str)
- ResponseBody: `BasicResponse(data=User)`（`models/data_models.py:5`）

5. #### 修改用户信息
- Path: `/v1/users/reset-info`
- Method: POST
- RequestBody: `RequestForUserChangeInfo`（`models/data_models.py:31`）
  - `user_name`(str)
  - `user_role`(str, 可选)
  - `status`(bool, 可选)
- ResponseBody: `BasicResponse`

6. #### 删除用户
- Path: `/v1/users/delete`
- Method: POST
- RequestParam:
  - `user_name`(str)
- ResponseBody: `BasicResponse`

7. #### 用户列表
- Path: `/v1/users/list`
- Method: POST
- RequestParam: 无
- ResponseBody: `BasicResponse(data=List[User])`

### 对话

1. #### 流式聊天（SSE）
- Path: `/v1/chat/`
- Method: POST
- RequestBody: `RequestForChat`
  - `content`(str)
  - `user_name`(str)
  - `session_id`(str, 可选)
  - `task`(str, 可选)
  - `tools`(list, 可选)
- Response: `text/event-stream

2. #### 初始化会话
- Path: `/v1/chat/init`
- Method: POST
- RequestParam: 无
- ResponseBody:
```
{
  "code": 200,
  "message": "ok",
  "data": { "session_id": "xxxx" }
}
```

3. #### 获取会话
- Path: `/v1/chat/get`
- Method: POST
- RequestParam:
  - `session_id`(str)
- ResponseBody: `BasicResponse(data=ChatSession|List[ChatMessage])`

4. #### 删除会话
- Path: `/v1/chat/delete`
- Method: POST
- RequestParam:
  - `session_id`(str)
- ResponseBody: `BasicResponse`

5. #### 列出会话
- Path: `/v1/chat/list`
- Method: POST
- RequestParam:
  - `user_name`(str)
- ResponseBody: `BasicResponse(data=List[ChatSession])`

6. #### 消息反馈
- Path: `/v1/chat/feedback`
- Method: POST
- RequestParam:
  - `session_id`(str)
  - `message_id`(int)
  - `feedback`(bool)
- ResponseBody: `BasicResponse`

### 股票收藏

1. #### 获取自选股
- Path: `/v1/stock/list_fav_stock`
- Method: POST
- RequestParam:
  - `user_name`(str)
- ResponseBody: `BasicResponse(data=List[StockFavInfo])`

2. #### 删除自选股
- Path: `/v1/stock/del_fav_stock`
- Method: POST
- RequestParam:
  - `user_name`(str)
  - `stock_code`(str)
- ResponseBody: `BasicResponse`

3. #### 添加自选股
- Path: `/v1/stock/add_fav_stock`
- Method: POST
- RequestParam:
  - `user_name`(str)
  - `stock_code`(str)
- ResponseBody: `BasicResponse`

4. #### 清空自选股
- Path: `/v1/stock/clear_fav_stock`
- Method: POST
- RequestParam:
  - `user_name`(str)
- ResponseBody: `BasicResponse`
