##2、chatbi的项目梳理一下，现在前端和后端交互接口有多少个，写一个接口文档

##1.1 用户管理接口 (/v1/users)  
#用户登录
+接口地址: /v1/users/login
+请求方法: POST
+请求参数: 
+  {
+    "user_name": "用户名",
+    "password": "密码"
+  }
+  
+  响应格式:
+  {
+    "code": 200,
+    "message": "用户登陆成功",
+    "data": []
+  }
+
#用户注册
+接口地址: /v1/users/register
+请求方法: POST
+请求参数:
+  {
+    "user_name": "用户名",
+    "password": "密码",
+    "user_role": "角色"
+  }
+
#重置密码
+接口地址: /v1/users/reset-password
+请求方法: POST
+请求参数:
+   {
+    "user_name": "用户名",
+    "password": "原密码",
+    "new_password": "新密码"
+  } 
+  
#获取用户信息
+接口地址: /v1/users/info
+请求方法: POST
+请求参数: user_name (查询参数)
+修改用户信息
+接口地址: /v1/users/reset-info
+请求方法: POST
+请求参数:
+  {
+    "user_name": "用户名",
+    "user_role": "角色(可选)",
+    "status": "状态(可选)"
+  }
+
#删除用户
+接口地址: /v1/users/delete
+请求方法: POST
+请求参数: user_name (查询参数)
+获取用户列表
+接口地址: /v1/users/list
+请求方法: POST
##1.2 股票管理接口 (/v1/stock)
#获取用户收藏股票
+接口地址: /v1/stock/list_fav_stock
+请求方法: POST
+请求参数: user_name (查询参数)
#删除用户收藏股票
+接口地址: /v1/stock/del_fav_stock
+请求方法: POST
+请求参数: user_name 和 stock_code (查询参数)
#添加用户收藏股票
+接口地址: /v1/stock/add_fav_stock
+请求方法: POST
+请求参数: user_name 和 stock_code (查询参数)
#清空用户收藏股票
+接口地址: /v1/stock/clear_fav_stock
+请求方法: POST
+请求参数: user_name (查询参数)
##1.3 聊天接口 (/v1/chat)
#发起聊天
+接口地址: /v1/chat/
+请求方法: POST
+请求参数:
+  {
+    "content": "用户提问",
+    "user_name": "用户名",
+    "session_id": "会话ID(可选)",
+    "task": "对话任务(可选)",
+    "tools": "工具列表(可选)"
+  }
+响应格式: 流式响应(SSE)
#初始化聊天
+接口地址: /v1/chat/init
+请求方法: POST
#获取聊天记录
+接口地址: /v1/chat/get
+请求方法: POST
+请求参数: session_id (查询参数)
#删除聊天记录
+接口地址: /v1/chat/delete
+请求方法: POST
+请求参数: session_id (查询参数)
#获取聊天列表
+接口地址: /v1/chat/list
+请求方法: POST
+请求参数: user_name (查询参数)
#反馈聊天质量
+接口地址: /v1/chat/feedback
+请求方法: POST
+请求参数: session_id, message_id, feedback (查询参数)
##1.4 数据管理接口 (/v1/data)
#下载数据
+接口地址: /v1/data/download
+请求方法: POST
#创建数据
+接口地址: /v1/data/create
+请求方法: POST
#上传数据
+接口地址: /v1/data/upload
+请求方法: POST
#删除数据
+接口地址: /v1/data/delete
+请求方法: POST
##1.5 股票数据接口 (/stock)
+这是底层股票数据API接口，挂载在主应用下：
#获取股票代码
+接口地址: /stock/get_stock_code
+请求方法: GET
+请求参数: keyword (查询参数，支持代码和名称模糊查询)
+获取指数代码
+接口地址: /stock/get_index_code
+请求方法: GET
#获取行业代码
+接口地址: /stock/get_industry_code
+请求方法: GET
#获取大盘信息
+接口地址: /stock/get_board_info
+请求方法: GET
#获取股票排行
+接口地址: /stock/get_stock_rank
+请求方法: GET
+请求参数:
+node: 股票市场/板块代码
+industryCode: 行业代码(可选)
+pageIndex: 页码(默认1)
+pageSize: 每页大小(默认100)
+sort: 排序字段(默认price)
+asc: 排序方式(0降序,1升序)
#获取月K线
+接口地址: /stock/get_month_line
+请求方法: GET
+请求参数: code(必需), startDate, endDate, type
#获取周K线
+接口地址: /stock/get_week_line
+请求方法: GET
+请求参数: code(必需), startDate, endDate, type
#获取日K线
+接口地址: /stock/get_day_line
+请求方法: GET
+请求参数: code(必需), startDate, endDate, type
#获取股票信息
+接口地址: /stock/get_stock_info
+请求方法: GET
+请求参数: code(必需)
#获取分时数据
+接口地址: /stock/get_stock_minute_data
+请求方法: GET
+请求参数: code(必需)
