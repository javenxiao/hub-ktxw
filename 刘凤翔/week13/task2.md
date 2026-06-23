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

## MCP 服务接口

### 股票数据服务 (AutoStock)
提供股票相关的实时数据查询：
- `GET /get_stock_code` - 查询股票代码
- `GET /get_index_code` - 查询指数代码
- `GET /get_industry_code` - 查询行业板块
- `GET /get_board_info` - 获取大盘数据
- `GET /get_stock_rank` - 股票排行
- `GET /get_month_line` - 月K线数据
- `GET /get_week_line` - 周K线数据
- `GET /get_day_line` - 日K线数据
- `GET /get_stock_info` - 股票基础信息
- `GET /get_stock_minute_data` - 分时数据

### 新闻服务 (News)
- `get_today_daily_news` - 今日新闻
- `get_douyin_hot_news` - 抖音热搜
- `get_github_hot_news` - GitHub热门
- `get_toutiao_hot_news` - 头条热点
- `get_sports_news` - 体育新闻

### 名言服务 (Saying)
- `get_today_familous_saying` - 名人名言
- `get_today_motivation_saying` - 励志语录
- `get_today_working_saying` - 工作鸡汤

### 工具服务 (Tools)
- `get_city_weather` - 城市天气
- `get_address_detail` - 地址解析
- `get_tel_info` - 手机号归属地
- `get_scenic_info` - 景点信息
- `get_flower_info` - 花语查询
- `get_rate_transform` - 汇率转换

## 前端页面

### Streamlit 用户界面
1. **user_login.py** - 用户登录页面
2. **user_register.py** - 用户注册页面
3. **user_info.py** - 用户信息页面
4. **user_list.py** - 用户列表页面（管理员）
5. **user_reset.py** - 用户信息修改页面

## 数据模型

### 主要数据表
- **user** - 用户表
- **user_favorite_stock** - 用户收藏股票表
- **chat_session** - 对话会话表
- **chat_message** - 对话消息表
- **data** - 数据表

## 技术架构

### 后端技术栈
- **框架**: FastAPI
- **数据库**: SQLite + SQLAlchemy ORM
- **AI 服务**: OpenAI Agents + MCP 协议
- **身份认证**: Session-based

### 前端技术栈
- **框架**: Streamlit
- **HTTP 客户端**: requests

### 特色功能
1. **智能对话**: 支持股票分析、数据BI、普通聊天等多种任务模式
2. **工具集成**: 通过 MCP 协议集成多种外部工具和服务
3. **流式响应**: 支持 SSE 流式对话响应
4. **会话管理**: 完整的对话历史记录和会话管理
5. **权限控制**: 基于角色的用户权限管理

---

**总结**: ChatBI 项目提供了完整的用户管理、智能对话、股票数据分析等功能，前后端接口设计清晰，支持多种业务场景。系统架构合理，具有良好的扩展性和可维护性。