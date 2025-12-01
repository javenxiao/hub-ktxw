import streamlit as st

from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

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

# 定义工具名称列表
NEWS_TOOLS = [
    "get_today_daily_news",
    "get_douyin_hot_news",
    "get_github_hot_news",
    "get_toutiao_hot_news",
    "get_sports_news"
]

TOOLS_TOOLS = [
    "get_city_weather",
    "get_address_detail",
    "get_tel_info",
    "get_scenic_info",
    "get_flower_info",
    "get_rate_transform",
    "sentiment_classification"
]

def detect_user_intent(prompt: str) -> str:
    """
    识别用户意图，返回工具类别
    返回: "news" - 新闻查询, "tools" - 工具调用, "general" - 通用对话
    """
    prompt_lower = prompt.lower()
    
    # 新闻相关关键词
    news_keywords = [
        "新闻", "news", "头条", "热点", "今日新闻", "daily news", 
        "抖音", "douyin", "github", "体育", "sports", "toutiao", "头条"
    ]
    
    # 工具相关关键词
    tools_keywords = [
        "天气", "weather", "地址", "address", "电话", "tel", "phone",
        "汇率", "exchange", "rate", "货币", "currency", "转换",
        "景点", "scenic", "旅游", "tourist", "花语", "flower",
        "情感", "sentiment", "分析", "analyze", "情感分析"
    ]
    
    # 检查是否包含新闻关键词
    for keyword in news_keywords:
        if keyword in prompt_lower:
            return "news"
    
    # 检查是否包含工具关键词
    for keyword in tools_keywords:
        if keyword in prompt_lower:
            return "tools"
    
    # 默认返回通用
    return "general"

async def get_model_response(prompt, model_name, use_tool):
    # 根据用户意图选择对应的工具过滤器
    intent = detect_user_intent(prompt)
    
    # 根据意图创建工具过滤器
    tool_filter = None
    if intent == "news":
        # 仅允许新闻工具
        tool_filter = {"allowed_tool_names": NEWS_TOOLS}
        intent_desc = "新闻查询模式"
    elif intent == "tools":
        # 仅允许功能工具
        tool_filter = {"allowed_tool_names": TOOLS_TOOLS}
        intent_desc = "工具调用模式"
    else:
        # 不过滤，使用所有工具
        tool_filter = None
        intent_desc = "通用模式（所有工具）"
    
    # 将意图信息存储到 session_state，供界面显示
    if use_tool:
        st.session_state['current_intent'] = intent_desc
    
    async with MCPServerSse(
            name="MCP-Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20,
            tool_filter=tool_filter
    )as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions="",
                mcp_servers=[mcp_server],
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
        
        # 显示当前使用的工具模式
        if use_tool and 'current_intent' in st.session_state:
            st.info(f"当前模式: {st.session_state['current_intent']}")

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
