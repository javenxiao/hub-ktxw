import streamlit as st
from agents.mcp.server import MCPServerSse, ToolFilterContext, MCPTool
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
# 导入自定义agents模块的核心类：
# Agent：智能代理（协调AI模型和工具调用）；Runner：运行代理的执行器；AsyncOpenAI：异步OpenAI兼容客户端；
# OpenAIChatCompletionsModel：ChatCompletion类型的AI模型封装；SQLiteSession：SQLite数据库会话（存储对话历史）

from openai.types.responses import ResponseTextDeltaEvent  # 导入OpenAI类型定义，用于识别流式响应的文本片段事件
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession("conversation_123")  # openai-agent提供的基于内存的上下文缓存

with st.sidebar:  # 创建页面侧边栏（用于放置配置项）
    st.title('职能AI+智能问答')  # 侧边栏标题

    # 检查会话状态中是否已保存API_TOKEN，且长度大于1（判断是否已配置Token）
    # session_state 保存当前的对话缓存
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')  # 已配置则显示成功提示
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    # 侧边栏添加密码输入框，用于输入API Token，默认值为已保存的key
    key = st.text_input('输入Token:', type='password', value=key)

    st.session_state['API_TOKEN'] = key  # 将输入的Token保存到会话状态（刷新页面不丢失）
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])  # 侧边栏添加下拉框，选择AI模型
    use_tool = st.checkbox("使用工具")  # 侧边栏添加复选框，控制是否启用工具调用（连接后端MCP服务）


# 初始化对话历史：如果会话状态中没有"messages"，则创建默认欢迎消息
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

# 渲染历史对话：遍历会话状态中的所有消息，按角色（user/assistant）显示
# streamlit提供的：st.session_state.messages此次对话的历史上下文
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # 按角色创建聊天消息框（用户/助手区分样式）
        st.write(message["content"])  # 显示消息内容


# 定义清空聊天历史的函数
def clear_chat_history():
    # 重置会话状态中的消息，恢复默认欢迎语
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

    global session  # 声明使用全局变量session
    session = SQLiteSession("conversation_123")  # 重新创建SQLite会话（清空对话数据库）


# 侧边栏添加"清空聊天"按钮，点击时调用clear_chat_history函数
st.sidebar.button('清空聊天', on_click=clear_chat_history)


async def tools_filter(context: ToolFilterContext, tool: MCPTool):
    user_input = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":  # 匹配用户角色的消息
            user_input = msg["content"].lower().strip()
            break

    if not user_input:
        return False
    else:
        print(f"从前端会话中获取的用户输入：{user_input}")

    tool_description = tool.description or ""
    if "[type: news]" in tool_description:
        tool_type = "news"
    elif "[type: tool]" in tool_description:
        tool_type = "tool"
    elif "[type: saying]" in tool_description:
        tool_type = "saying"
    else:
        tool_type = None

    if not tool_type:
        return False

    if any(keyword in user_input for keyword in ["今日新闻", "热门", "话题", "抖音", "GitHub", "今日头条", "电竞", "体育"]):
        is_match = tool_type == "news"
        if is_match:
            print(f"关键词匹配成功，{tool.name} 备选")
        else:
            print(f"关键词匹配失败，{tool.name} 过滤")
        return is_match
    elif any(keyword in user_input for keyword in ["天气", "地址", "电话", "旅游", "景点", "花", "货币", "汇率", "情感", "分析"]):
        is_match = tool_type == "tool"
        if is_match:
            print(f"关键词匹配成功，{tool.name} 备选")
        else:
            print(f"关键词匹配失败，{tool.name} 过滤")
        return is_match
    elif any(keyword in user_input for keyword in ["名言", "一言", "励志", "激励", "职场", "心灵鸡汤"]):
        is_match = tool_type == "saying"
        if is_match:
            print(f"关键词匹配成功，{tool.name} 备选")
        else:
            print(f"关键词匹配失败，{tool.name} 过滤")
        return is_match

    print(f"无匹配关键词，{tool.name} 过滤")
    return False


# 定义异步函数：获取AI模型响应（核心逻辑，包含MCP工具调用和AI交互）
async def get_model_response(prompt, model_name, use_tool):
    """
    :param prompt: 当前用户输入
    :param model_name: 模型版本
    :param use_tool: 是否调用工具
    :return:
    """
    # 异步创建MCP的SSE客户端，连接后端MCP服务器（之前主程序启动的8900端口SSE服务）
    async with MCPServerSse(
            name="SSE Python Server",  # MCP客户端名称
            params={
                "url": "http://localhost:8900/sse",  # 后端MCP服务器的SSE访问地址
            },
            client_session_timeout_seconds=20,
            tool_filter=tools_filter
    )as mcp_server:  # 上下文管理器自动管理MCP客户端的连接/关闭

        # 创建异步OpenAI兼容客户端（对接阿里云通义千问API）
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 根据是否启用工具，创建不同配置的Agent（智能代理）
        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions="",   # 代理的系统指令
                mcp_servers=[mcp_server],  # 关联MCP服务器（启用工具调用，可调用news/saying/tool模块的功能）
                model=OpenAIChatCompletionsModel(  # 配置AI模型
                    model=model_name,  # 选择的模型（qwen-flash/max）
                    openai_client=external_client,  # 绑定前面创建的通义千问客户端
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                # 不关联MCP服务器（仅纯AI对话，不调用任何工具）
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        # 运行Agent，以流式方式处理用户输入（prompt），并关联对话会话（session，openai-agent中定义的保存历史上下文）
        result = Runner.run_streamed(agent, input=prompt, session=session)
        # 异步遍历流式响应事件
        async for event in result.stream_events():
            # 筛选出"原始响应事件"，且数据是文本片段事件（ResponseTextDeltaEvent）
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta  # 生成器返回当前文本片段（实现流式输出）

# 只有当用户输入的API Token长度大于1（已配置有效Token）时，才允许发送聊天请求
if len(key) > 1:
    # 监听用户的聊天输入框（st.chat_input()创建底部输入框，用户输入后返回内容）
    if prompt := st.chat_input():
        # 将用户输入添加到会话状态的消息列表中
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 渲染用户输入的消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 渲染助手的响应消息框
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # 创建空占位符，用于动态更新响应内容
            full_response = ""  # 存储完整的响应文本

            with st.spinner("请求中..."):  # 显示加载中提示
                try:
                    # 调用get_model_response获取流式响应生成器
                    response_generator = get_model_response(prompt, model_name, use_tool)

                    # 定义内部异步函数：遍历流式生成器，累积文本并更新界面
                    async def stream_and_accumulate(generator):
                        accumulated_text = ""
                        async for chunk in generator:   # 异步遍历每个文本片段
                            accumulated_text += chunk  # 累积文本
                            # 在占位符中显示已累积的文本，并添加"▌"光标效果
                            message_placeholder.markdown(accumulated_text + "▌")
                        return accumulated_text  # 返回完整文本

                    # 运行异步函数，获取完整响应
                    full_response = asyncio.run(stream_and_accumulate(response_generator))
                    # 响应完成后，移除光标，显示完整文本
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print(f"Error during streaming: {e}")

            # 将完整的助手响应添加到会话状态的消息列表中（保存历史）
            st.session_state.messages.append({"role": "assistant", "content": full_response})