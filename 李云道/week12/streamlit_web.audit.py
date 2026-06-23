import asyncio
import streamlit as st
from agents import (
    SQLiteSession,
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    set_default_openai_api,
    set_tracing_disabled,
    ModelSettings
)
from agents.mcp import (
    MCPServerSse,
    ToolFilterStatic,
    ToolFilterCallable
)
from openai import AsyncClient
from openai.types.responses import ResponseTextDeltaEvent

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession(__name__)


def init_chat():
    return [{"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话，也可以调用内部工具。"}]


def clear_chat_his():
    st.session_state.messages = init_chat()
    global session
    session = SQLiteSession(__name__)


def check_api_key(api_key):
    return len(api_key) > 1


toos_tool_list = ['get_city_weather', 'get_address_detail', 'get_tel_info',
                  'get_scenic_info', 'get_flower_info', 'get_rate_transform']


def tools_mcps_callable_filter(context, tool) -> bool:
    return tool.name in toos_tool_list


saying_tool_list = ['get_today_familous_saying', 'get_today_motivation_saying', 'get_today_working_saying']


def saying_mcps_callable_filter(context, tool) -> bool:
    return tool.name in saying_tool_list


sentiment_tool_list = ['sentiment_classification', 'sentiment_comparison', 'emotional_trend_analysis']


def sentiment_mcps_callable_filter(context, tool) -> bool:
    return tool.name in sentiment_tool_list


async def get_model_response_generator(model_name, prompt, use_tool):
    client = AsyncClient(
        api_key=key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if use_tool:
        # news 静态工具 ToolFilterStatic
        news_tool_list = ['get_today_daily_news', 'get_douyin_hot_news', 'get_github_hot_news',
                          'get_toutiao_hot_news', 'get_sports_news']
        news_mcps_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=news_tool_list)
        news_server = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            client_session_timeout_seconds=20,
            tool_filter=news_mcps_filter
        )

        # tools 动态工具 ToolFilterCallable
        tools_server = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            client_session_timeout_seconds=20,
            tool_filter=tools_mcps_callable_filter
        )
        saying_server = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            client_session_timeout_seconds=20,
            tool_filter=saying_mcps_callable_filter
        )
        sentiment_server = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            client_session_timeout_seconds=20,
            tool_filter=sentiment_mcps_callable_filter
        )

        async with news_server, tools_server, saying_server, sentiment_server:
            # 定义 MCP tool Agent
            news_agent = Agent(
                name="news_assistant",
                instructions="您是一个新闻助手，专门处理新闻相关的查询，包括今日新闻、抖音热点、GitHub热门、头条热点和体育新闻。",
                mcp_servers=[news_server],
                model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            tools_agent = Agent(
                name="tools_assistant",
                instructions="您是一个工具助手，专门处理计算、天气、时间转换和单位转换等工具类任务。",
                mcp_servers=[tools_server],
                model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            saying_agent = Agent(
                name="saying_assistant",
                instructions="您是一个名言助手，专门提供名言警句和励志语录。",
                mcp_servers=[saying_server],
                model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            sentiment_agent = Agent(
                name="sentiment_assistant",
                instructions="您是一个情感分析助手，专门处理文本情感分析和情绪检测任务。",
                mcp_servers=[sentiment_server],
                model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            # 创建路由agent
            agent = Agent(
                name="routing_agent",
                instructions="Handoff to the appropriate agent based on the language of the request.",
                handoffs=[news_agent, tools_agent, saying_agent, sentiment_agent],
                model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            # 创建handoff步骤记录
            if 'handoff_steps' not in st.session_state:
                st.session_state.handoff_steps = []

            # 清空之前的记录
            st.session_state.handoff_steps = []

            # 流式运行并捕获所有事件
            result = Runner.run_streamed(agent, input=prompt, session=session)

            # 用于收集文本响应的变量
            full_text = ""

            # 处理所有事件
            async for event in result.stream_events():
                # 记录handoff相关事件
                if hasattr(event, 'type') and hasattr(event, 'data'):
                    if event.type != "raw_response_event":
                        event_info = {
                            'type': event.type,
                            'timestamp': asyncio.get_event_loop().time(),
                            'data': str(event.data)
                        }
                        st.session_state.handoff_steps.append(event_info)

                    # 打印到控制台用于调试
                    # print(f"Event: {event.type} - Data: {event_info['data']}")

                # 处理文本响应事件
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    full_text += delta
                    yield delta

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

    if "API_KEY" in st.session_state and len(st.session_state["API_KEY"]) > 1:
        st.success("API_KEY 已经配置", icon='✅')
        key = st.session_state["API_KEY"]
    else:
        key = ""

    key = st.text_input("请输入Token", type="password", value=key)
    st.session_state["API_KEY"] = key
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")

    st.button("清空聊天", on_click=clear_chat_his)

# 初始化对话
if "messages" not in st.session_state:
    st.session_state.messages = init_chat()

# 显示对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 显示handoff步骤的展开面板（在聊天前显示）
if st.session_state.get('handoff_steps'):
    with st.expander("Handoff执行步骤详情", expanded=False):
        st.markdown("### 路由决策过程")
        for i, step in enumerate(st.session_state.handoff_steps):
            with st.expander(f"步骤 {i + 1}: {step['type']}", expanded=False):
                st.json(step, expanded=False)

# 聊天功能
if check_api_key(key):
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.empty()
            model_response = ""

        with st.spinner("请求中..."):
            try:
                print(model_name, prompt, use_tool)
                response_generator = get_model_response_generator(model_name, prompt, use_tool)


                async def stream2text(generator):
                    full_text = ""
                    async for chunk in generator:
                        full_text += chunk
                        response.markdown(full_text + "▌")
                    return full_text


                model_response = asyncio.run(stream2text(response_generator))
                response.markdown(model_response)

                # 显示handoff步骤总结
                if use_tool and st.session_state.get('handoff_steps'):
                    st.success(f"✅ Handoff过程完成，共{len(st.session_state.handoff_steps)}个步骤")

            except Exception as e:
                error_message = f"发生错误: {e}"
                response.error(error_message)
                print(f"Error get_model_response: {e}")

        st.session_state.messages.append({"role": "assistant", "content": model_response})

        # 强制重新渲染以确保handoff步骤立即显示
        st.rerun()