import streamlit as st
from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from agents import set_default_openai_api, set_tracing_disabled

tool_names = ['get_today_daily_news', 'get_douyin_hot_news', 'get_github_hot_news', 'get_toutiao_hot_news',
              'get_sports_news', 'get_today_familous_saying', 'get_today_motivation_saying', 'get_today_working_saying',
              'get_city_weather', 'get_address_detail', 'get_tel_info', 'get_scenic_info', 'get_flower_info',
              'get_rate_transform', 'sentiment_classification']

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession("conversation_123")

with st.sidebar:
    st.title('职能AI+智能问答')
    print("st.session_state:", st.session_state)
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    key = st.text_input('输入Token:', type='password', value=key)

    st.session_state['API_TOKEN'] = key
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])

    # 工具选择
    tool_options = ["不使用工具"] + tool_names  # 添加不使用工具选项
    selected_tool_option = st.selectbox("选择工具", tool_options)

    if selected_tool_option == "不使用工具":
        use_tool = None
    else:
        use_tool = selected_tool_option

# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]
    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)


async def get_model_response(prompt, model_name, use_tool):
    try:
        print(f"开始处理请求: {prompt}")
        print(f"选择的工具: {use_tool}")

        async with MCPServerSse(
                name="SSE Python Server",
                params={
                    "url": "http://localhost:3006/sse",
                },
                client_session_timeout_seconds=30  # 增加超时时间
        ) as mcp_server:
            print("MCP服务器连接成功")

            external_client = AsyncOpenAI(
                api_key=key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=30.0  # 添加超时设置
            )
            print("OpenAI客户端创建成功")

            # 创建Agent的基础参数
            agent_kwargs = {
                "name": "Assistant",
                "model": OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            }

            if use_tool:
                # 修复1: mcp_servers 应该是一个列表
                agent_kwargs["mcp_servers"] = [mcp_server]
                # 修复2: instructions 拼写错误
                agent_kwargs["instructions"] = f"""
                你是一个有帮助的助手。当用户的问题适合使用工具"{use_tool}"时，请使用这个工具来帮助用户。
                如果不适合使用这个工具，请直接回答问题。"""
                print(f"创建带工具的Agent，使用工具: {use_tool}")
            else:
                agent_kwargs["instructions"] = "你是一个有帮助的助手，请直接回答用户的问题。"
                print("创建纯对话Agent")

            print("创建Agent...")
            agent = Agent(**agent_kwargs)

            print("开始执行Runner...")
            result = await Runner.run(agent, input=prompt, session=session)
            print("Runner执行成功")

            return result.final_output

    except Exception as e:
        print(f"get_model_response错误: {e}")
        import traceback
        traceback.print_exc()
        return f"请求失败: {str(e)}。请确保MCP服务器正在运行在端口3006。"


if len(key) > 1:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("请求中..."):
                try:
                    # 直接获取响应，不使用流式处理
                    full_response = asyncio.run(get_model_response(prompt, model_name, use_tool))
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print(f"Error: {e}")
                    import traceback

                    traceback.print_exc()

            # 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
