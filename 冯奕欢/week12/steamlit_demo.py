import asyncio

import streamlit as st
from agents import SQLiteSession, Agent, Runner
from agents.mcp.server import MCPServerSse
from openai.types.responses import ResponseTextDeltaEvent

from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

import os
os.environ["OPENAI_API_KEY"] = "sk-3bbac92a5f504fd7b7b38dbe4f1a11e4"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from global_data import get_session, reset_session

st.set_page_config(page_title="企业职能机器人")

with st.sidebar:
    # 标题
    st.title('职能AI+智能问答')
    # 用户输入框
    if 'USER' in st.session_state:
        user = st.session_state['USER']
    else:
        user = "123"
    user = st.text_input(label="用户", value=user, type='default')
    # API Token 输入框
    if 'OPEN_KEY' in st.session_state:
        key = st.session_state['OPEN_KEY']
    else:
        key = ""
    key = st.text_input(label="API Token：", value=key, type="password")
    st.session_state['OPEN_KEY'] = key
    # 大模型选择框
    model_name = st.selectbox(label="选择模型：", options=["qwen-flash", "qwen-max"])
    # 是否使用工具
    use_tool = st.checkbox(label="使用工具")
    # 是否过滤工具
    use_tool_filter = st.checkbox(label="Tool Filter")

if len(key) > 1:
    st.sidebar.success("API Token已经配置")

def clear_chat_history():
    """清空历史聊天记录"""
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话，也可以调用内部工具。"}
    ]
    reset_session(user)


# 清空历史聊天记录按钮
st.sidebar.button(label="清空聊天消息", on_click=clear_chat_history)


def get_custom_tool_filter(prefix):
    """返回ToolFileter"""
    def custom_tool_filter(context, tool) -> bool:
        # 根据工具的前缀判断是否过滤
        result = tool.name.startswith(prefix)
        if result:
            print("user tool ==> ", tool.name)
        return result
    return custom_tool_filter


async def get_model_response(prompt, model_name, use_tool, session):
    """
    从大模型获取回答
    :param prompt: 提示词
    :param model_name: 模型名称
    :param use_tool: 是否使用工具
    :return: 大模型回答结果
    """
    if use_tool_filter:
        # 查询需要使用工具前缀
        agent = Agent(
            name="工具选择助手",
            instructions="""
你擅长根据文本内容分析出可能需要的工具，有如下工具：
- 查询新闻：news
- 天气、路线、地址、汇率等工具：tool
- 名言名句：saying
- 情感分类：classification
返回内容只能是以上工具的其中一个，如查询今日头像新闻，返回news。如果不需要以上的工具，请返回none。
    """,
            model=model_name
        )
        tool = await Runner.run(agent, prompt)
        prefix = tool.final_output
        # 创建工具过滤器
        if tool == 'none':
            tool_filter = None
        else:
            tool_filter = get_custom_tool_filter(prefix)
    else:
        tool_filter = None
    async with MCPServerSse(
        name="MCP Sse Server",
        params={
            "url": "http://localhost:8900/sse"
        },
        tool_filter=tool_filter,
        client_session_timeout_seconds=20
    ) as mcp_server:
        # 创建 Agent
        if use_tool:
            # 使用工具
            agent = Agent(
                name="Assistant",
                instructions="",
                mcp_servers=[mcp_server],
                model=model_name
            )
        else:
            # 不使用工具
            agent = Agent(
                name="Assistant",
                instructions="",
                model=model_name
            )
        # 大模型调用 流式回答
        print("session-->", session)
        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


async def stream_and_accumulate(generator):
    """
    流式回答内容
    :param generator: 生成器
    :return: 生成器回答的结果
    """
    accumulated_text = ""
    async for chunk in generator:
        accumulated_text += chunk
        message_placeholder.markdown(accumulated_text + "▌")
    return accumulated_text


# 初始化消息列表
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话，也可以调用内部工具。"}
    ]
# 遍历消息显示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 输入框
if len(key) > 1:
    # 有 API TOKEN才显示输入框
    if prompt := st.chat_input():
        # 插入用户问题
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # 插入系统回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # 状态
            with st.spinner("请求中..."):
                try:
                    # 从大模型获取回答
                    response_generator = get_model_response(prompt, model_name, use_tool, get_session(user))
                    full_response = asyncio.run(stream_and_accumulate(response_generator))
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    # 异常处理
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
            # 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})