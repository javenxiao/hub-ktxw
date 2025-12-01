import streamlit as st

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

def filter_tools_by_intent(user_input, all_tools):
    """
    根据用户输入意图过滤工具
    """
    user_input_lower = user_input.lower()

    # 定义工具分类关键词
    news_keywords = ['新闻', '热点', '头条', '日报', '资讯', '消息', '报道']
    saying_keywords = ['名言', '语录', '格言', '警句', '励志', '鸡汤', '鼓励']
    tool_keywords = [
        '天气', '气温', '气象', '气候',
        '电话', '手机', '号码', '归属',
        '地址', '位置', '地点', '解析',
        '景点', '旅游', '景区', '风景',
        '花语', '鲜花', '花',
        '汇率', '货币', '兑换', '换钱',
        '情感', '情绪', '心情', '感情', '分析'
    ]

    # 判断用户意图
    is_news_related = any(keyword in user_input_lower for keyword in news_keywords)
    is_saying_related = any(keyword in user_input_lower for keyword in saying_keywords)
    is_tool_related = any(keyword in user_input_lower for keyword in tool_keywords)

    # 如果没有明确意图，使用所有工具
    if not (is_news_related or is_saying_related or is_tool_related):
        return all_tools

    # 根据意图过滤工具
    filtered_tools = []
    for tool in all_tools:
        tool_name_lower = tool.name.lower()

        if is_news_related and 'news' in tool_name_lower:
            filtered_tools.append(tool)
        elif is_saying_related and 'saying' in tool_name_lower:
            filtered_tools.append(tool)
        elif is_tool_related and any(keyword in tool_name_lower for keyword in
                                     ['weather', 'tel', 'address', 'scenic', 'flower', 'rate', 'sentiment']):
            filtered_tools.append(tool)

    return filtered_tools

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession("conversation_123")

with st.sidebar:
    st.title('职能AI+智能问答')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    key = st.text_input('输入Token:', type='password', value=key)

    st.session_state['API_TOKEN'] = key
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")


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
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        if use_tool:
            # 获取所有可用工具
            all_tools = await mcp_server.list_tools()

            # 根据用户输入动态过滤工具
            filtered_tools = filter_tools_by_intent(prompt, all_tools)

            # 如果过滤后没有工具，使用所有工具作为后备
            if not filtered_tools:
                filtered_tools = all_tools

            agent = Agent(
                name="Assistant",
                instructions="根据用户的问题选择合适的工具来回答。",
                tools=filtered_tools,  # 使用过滤后的工具列表
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta

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
                    response_generator = get_model_response(prompt, model_name, use_tool)

                    async def stream_and_accumulate(generator):
                        accumulated_text = ""
                        async for chunk in generator:
                            accumulated_text += chunk
                            message_placeholder.markdown(accumulated_text + "▌")
                        return accumulated_text

                    full_response = asyncio.run(stream_and_accumulate(response_generator))
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print(f"Error during streaming: {e}")

            # 4. 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
