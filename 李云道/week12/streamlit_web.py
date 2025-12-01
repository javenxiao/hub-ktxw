import asyncio
import streamlit as st
from agents import (
    SQLiteSession,
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_default_openai_api,
    set_tracing_disabled
)
from agents.mcp import MCPServerSse
from openai import AsyncClient
from openai.types.responses import ResponseTextDeltaEvent

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession(__name__)

TOOL_CATEGORIES = {
    "新闻服务": {
        "description": "使用新闻相关工具",
        "url": "http://localhost:8900/sse"
    },
    "名言服务": {
        "description": "使用名言相关工具",
        "url": "http://localhost:8901/sse"
    },
    "工具服务": {
        "description": "使用各种实用工具",
        "url": "http://localhost:8902/sse"
    },
    "情感服务": {
        "description": "使用情感分析工具",
        "url": "http://localhost:8903/sse"
    },
    "所有服务": {
        "description": "使用所有可用服务",
        "url": "all"
    }
}


def init_chat():
    '''
    聊天初始化
    :return:
    '''
    return [{"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话，也可以调用内部工具。"}]


def clear_chat_his():
    '''
    清空聊天记录
    :return:
    '''
    st.session_state.messages = init_chat()

    global session
    session = SQLiteSession(__name__)


def check_api_key(api_key):
    '''
    检查api_key
    :param api_key:
    :return:
    '''
    if len(api_key) > 1:
        return True
    return False


async def get_model_response_generator(model_name, prompt, use_tool, tool):
    client = AsyncClient(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 根据 use_tool 参数选择不同的逻辑分支
    if use_tool and tool in TOOL_CATEGORIES:
        # 使用 所有 MCP 工具
        if TOOL_CATEGORIES[tool]["url"] == "all":
            mcp_servers = []
            for k, v in TOOL_CATEGORIES.items():
                if v["url"] == "all":
                    continue
                mcp_server = MCPServerSse(
                    name="SSE Python Server",
                    params={
                        "url": v["url"],
                    },
                    client_session_timeout_seconds=20
                )
                mcp_servers.append(mcp_server)
        # 使用单个 MCP 工具
        else:
            # print(TOOL_CATEGORIES[tool]["url"])
            mcp_servers = [
                MCPServerSse(
                    name="SSE Python Server",
                    params={
                        "url": TOOL_CATEGORIES[tool]["url"],
                    },
                    client_session_timeout_seconds=20
            )]
        # 定义 MCP Agent
        agent = Agent(
            name="assistant",
            instructions="You are a helpful assistant.",
            mcp_servers=mcp_servers,
            model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
        )

        # 流式运行
        result = Runner.run_streamed(agent, input=prompt, session=session)
        # 流式生成器
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta
    else:
        # 不使用 MCP 工具
        agent = Agent(
            name="assistant",
            instructions="",
            model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
        )
        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


# 侧边栏
with st.sidebar:
    st.title("职能+智能 AI")

    # session缓存
    if "API_KEY" in st.session_state and len(st.session_state["API_KEY"]) > 1:
        st.success("API_KEY 已经配置", icon='✅')
        key = st.session_state["API_KEY"]
    else:
        key = ""

    # 侧边栏组件
    key = st.text_input("请输入Token", type="password", value=key)
    st.session_state["API_KEY"] = key
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")
    if use_tool:
        tool = st.selectbox("选择工具", list(TOOL_CATEGORIES.keys()), index=0)
    else:
        tool = "未使用"

    st.button("清空聊天", on_click=clear_chat_his)

# 初始化对话
if "messages" not in st.session_state:
    st.session_state.messages = init_chat()

# 显示对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 聊天功能
if check_api_key(key):

    # 海象运算符，获取用户输入赋值给prompt，并判断
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户输入消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 显示回答
        with st.chat_message("assistant"):
            response = st.empty()  # 创建空占位符
            model_response = ""

        with st.spinner("请求中..."):
            try:
                print(model_name, prompt, use_tool, tool)
                response_generator = get_model_response_generator(model_name, prompt, use_tool, tool)


                async def stream2text(generator):
                    full_text = ""
                    async for chunk in generator:
                        full_text += chunk
                        response.markdown(full_text + "▌")  # 动态全量更新，带光标效果
                    return full_text


                model_response = asyncio.run(stream2text(response_generator))
                response.markdown(model_response)

            except Exception as e:
                error_message = f"发生错误: {e}"
                response.error(error_message)
                print(f"Error get_model_response: {e}")

        # 消息缓存
        st.session_state.messages.append({"role": "assistant", "content": model_response})
