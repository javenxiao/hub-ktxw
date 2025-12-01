## 📋 项目概述
这是一个基于MCP协议的企业智能助手系统，包含MCP服务器、工具集成和Streamlit前端界面。

---

## 🛠️ 环境配置步骤

### 步骤1：创建专用Python环境
```bash
# 激活新环境
conda conda activate py312
```

### 步骤2：安装项目依赖
```bash
# 安装基础依赖包
pip install fastmcp streamlit fastapi uvicorn requests httpx -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 安装openai-agents（AI对话功能需要）
pip install git+https://github.com/openai/openai-agents-python.git -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤3：验证安装
```bash
python -c "
import fastmcp; print('✅ fastmcp 安装成功')
import streamlit; print('✅ streamlit 安装成功') 
import fastapi; print('✅ fastapi 安装成功')
print('环境配置完成！')
"
```

---

## 🚀 项目启动步骤

### 步骤4：启动MCP服务器（终端1）
```bash
# 进入项目目录
cd 项目根目录/mcp_server

# 启动MCP服务器
python mcp_server_main.py
```
**预期输出**：
- 服务器启动在 `http://localhost:8900`
- 显示可用的工具列表
- 持续运行，不要关闭此终端

### 步骤5：验证MCP服务器（终端2 - 可选测试）
```bash
# 在新终端中，确保已激活相同环境
conda activate mcp-agent
cd 项目根目录/mcp_server

# 测试基础功能
python mcp_client_test1.py
```
**预期输出**：显示工具列表和API调用结果

### 步骤6：启动Streamlit前端（终端3）
```bash
# 在新终端中，确保已激活相同环境
conda activate mcp-agent
cd 项目根目录

# 启动Streamlit界面
streamlit run steamlit_demo.py
```
**预期输出**：
- Streamlit服务启动在 `http://localhost:8501`
- 浏览器自动打开应用界面

---

## ⚙️ 界面配置步骤

### 步骤7：配置Streamlit界面
在浏览器中打开的Streamlit界面中：

1. **输入API Token**：
   ```
   sk-78cc4e9ac8f44efdb207b7232e1ae6d8
   ```

2. **选择模型**：`qwen-flash` 或 `qwen-max`

3. **勾选"使用工具"**：启用MCP工具调用功能

4. **开始对话**：在底部输入框提问

---

## 🔧 故障排除

### 如果依赖安装失败
```bash
# 清除缓存重试
pip cache purge

# 逐个安装包
pip install 包名 -i https://pypi.org/simple/
```

### 如果端口被占用
```bash
# 查找占用端口的进程
lsof -i :8900
lsof -i :8501

# 终止进程
kill -9 <进程ID>
```

### 如果MCP服务器连接失败
- 确保MCP服务器在终端1中持续运行
- 检查端口8900是否被占用
- 验证服务器启动时没有错误信息

### 如果Streamlit界面无响应
- 检查Token是否正确输入
- 确认MCP服务器正在运行
- 查看浏览器控制台错误信息（F12）

---

## ✅ 验证完整功能

成功启动后，您可以测试以下功能：

1. **基础对话**：在Streamlit中输入普通问题
2. **工具调用**：尝试以下问题：
   - "今天有什么新闻？"
   - "北京的天气怎么样？"
   - "给我一句励志名言"
3. **多轮对话**：进行连续提问

---

## 📝 重要提醒

1. **保持三个终端同时运行**：
   - 终端1：MCP服务器
   - 终端2：测试客户端（可选）
   - 终端3：Streamlit前端

2. **API密钥**：使用项目中提供的密钥或申请自己的阿里云API密钥

3. **网络连接**：确保可以访问外部API服务

按照这个完整的步骤操作，您应该能够成功启动整个项目。如果在任何步骤遇到问题，请告诉我具体的错误信息，我会帮您解决！