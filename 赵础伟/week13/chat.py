import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from typing import AsyncGenerator

from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from jinja2 import Environment, FileSystemLoader

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable
from fastapi.responses import StreamingResponse

# 新增：定义agent类型
class AgentType:
    CHAT = "chat"
    STOCK = "stock"

def generate_random_chat_id(length=12):
    with SessionLocal() as session:
        for retry_time in range(20):
            characters = string.ascii_letters + string.digits
            session_id = ''.join(random.choice(characters) for i in range(length))
            chat_session_record: ChatSessionTable | None = session.query(ChatSessionTable).filter(
                ChatSessionTable.session_id == session_id).first()
            if chat_session_record is None:
                break

            if retry_time > 10:
                raise Exception("Failed to generate a unique session_hash")

    return session_id

def get_agent_instructions(agent_type: str, task: str = None) -> str:
    """根据agent类型获取对应的系统提示"""
    env = Environment(loader=FileSystemLoader("templates"))

    if agent_type == AgentType.STOCK:
        template = env.get_template("stock_agent_system_prompt.jinjia2")
        task_description = """
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
5. 当用户的问题与股票无关时，应该主动将对话转交给闲聊agent。
6. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
"""
    else:  # 默认闲聊agent
        template = env.get_template("chat_agent_system_prompt.jinjia2")
        task_description = """
1. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
2. 避免过于专业或生硬的术语，除非用户明确要求。
3. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
4. 当用户询问股票、投资、金融相关的问题时，应该主动将对话转交给股票agent。
5. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
6. 关键词：友好、轻松、富有同理心。
        """

    system_prompt = template.render(
        agent_name="小呆助手",
        task_description=task_description,
        current_datetime=datetime.now(),
        agent_type=agent_type
    )
    return system_prompt

def detect_agent_switch(current_agent: str, user_input: str) -> str:
    """检测是否需要切换agent"""
    user_input_lower = user_input.lower()

    # 股票相关关键词
    stock_keywords = [
        '股票', '股价', '股市', '投资', '基金', '证券', 'k线', '涨停', '跌停',
        '大盘', '板块', '行情', '走势', '分析', '估值', 'pe', 'pb', 'roe',
        '上证', '深证', '创业板', '科创板', '港股', '美股', 'a股',
        'sh', 'sz', 'hk', 'us', 'nasdaq', 'nyse'
    ]

    # 闲聊相关关键词（当股票agent检测到这些时转交）
    chat_keywords = [
        '天气', '新闻', '笑话', '故事', '闲聊', '聊天', '你好', '嗨',
        '心情', '情感', '生活', '日常', '娱乐', '音乐', '电影', '体育'
    ]

    if current_agent == AgentType.CHAT:
        # 如果当前是闲聊agent，检测到股票关键词就切换到股票agent
        if any(keyword in user_input_lower for keyword in stock_keywords):
            return AgentType.STOCK
    elif current_agent == AgentType.STOCK:
        # 如果当前是股票agent，检测到纯粹闲聊内容就切换到闲聊agent
        if (not any(keyword in user_input_lower for keyword in stock_keywords) and
            any(keyword in user_input_lower for keyword in chat_keywords)):
            return AgentType.CHAT

    return current_agent  # 不需要切换

def get_agent_tools(agent_type: str) -> List[str]:
    """根据agent类型获取对应的工具列表"""
    if agent_type == AgentType.STOCK:
        # 股票agent的工具
        return [
            "get_stock_code", "get_index_code", "get_industry_code",
            "get_board_info", "get_stock_rank", "get_month_line",
            "get_week_line", "get_day_line", "get_stock_info",
            "get_stock_minute_data"
        ]
    else:  # 闲聊agent的工具
        return [
            "get_today_daily_news", "get_douyin_hot_news", "get_github_hot_news",
            "get_toutiao_hot_news", "get_sports_news", "get_today_familous_saying",
            "get_today_motivation_saying", "get_today_working_saying", "get_city_weather",
            "get_address_detail", "get_tel_info", "get_scenic_info", "get_flower_info",
            "get_rate_transform"
        ]

def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        task: str,
) -> str:
    # 创建对话的title，通过summary agent
    # 存储数据库
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()

        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question,
        )
        print("add ChatSessionTable", user_id[0], session_id)
        session.add(chat_session_record)
        session.commit()
        session.flush()

        # 初始化为闲聊agent
        initial_agent = AgentType.CHAT
        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=get_agent_instructions(initial_agent, task)
        )
        session.add(message_recod)
        session.flush()
        session.commit()

    return True

async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    # 对话管理，通过session id
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 获取当前agent类型（从数据库或默认）
    current_agent = await get_current_agent_type(session_id)

    # 检测是否需要切换agent
    new_agent = detect_agent_switch(current_agent, content)

    # 如果agent发生变化，添加切换提示
    if new_agent != current_agent:
        switch_message = f"\n【系统提示】检测到对话主题变化，正在切换到{'股票' if new_agent == AgentType.STOCK else '闲聊'}助手...\n"
        yield switch_message
        await update_agent_type(session_id, new_agent)
        current_agent = new_agent

    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)

    # 获取对应agent的system message
    instructions = get_agent_instructions(current_agent, task)

    # 获取对应agent的工具
    agent_tools = get_agent_tools(current_agent)
    if tools:  # 如果前端指定了工具，则使用指定的工具
        agent_tools = [tool for tool in tools if tool in agent_tools]

    # agent 初始化
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # mcp tools 选择
    if not agent_tools or len(agent_tools) == 0:
        tool_mcp_tools_filter: Optional[ToolFilterStatic] = None
    else:
        tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=agent_tools)

    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    # openai-agent支持的session存储
    session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 如果没有选择工具，默认直接调用大模型回答
    if not agent_tools or len(agent_tools) == 0:
        agent = Agent(
            name=f"{'股票' if current_agent == AgentType.STOCK else '闲聊'}助手",
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            ),
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

        result = Runner.run_streamed(agent, input=content, session=session)

        assistant_message = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    if event.data.delta:
                        yield f"{event.data.delta}"
                        assistant_message += event.data.delta

        append_message2db(session_id, "assistant", assistant_message)

    # 需要调用mcp 服务进行回答
    else:
        async with mcp_server:
            # 哪些工具直接展示结果
            need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
            if set(need_viz_tools) & set(agent_tools):
                tool_use_behavior = "stop_on_first_tool"
            else:
                tool_use_behavior = "run_llm_again"

            agent = Agent(
                name=f"{'股票' if current_agent == AgentType.STOCK else '闲聊'}助手",
                instructions=instructions,
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=os.environ["OPENAI_MODEL"],
                    openai_client=external_client,
                ),
                tool_use_behavior=tool_use_behavior,
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            result = Runner.run_streamed(agent, input=content, session=session)

            assistant_message = ""
            current_tool_name = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        current_tool_name = event.data.item.name
                        yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                        assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta
                    assistant_message += event.data.delta

            append_message2db(session_id, "assistant", assistant_message)

async def get_current_agent_type(session_id: str) -> str:
    """从数据库获取当前agent类型"""
    with SessionLocal() as session:
        # 这里需要修改数据库模型来存储agent类型
        # 暂时使用默认值
        return AgentType.CHAT

async def update_agent_type(session_id: str, agent_type: str) -> bool:
    """更新数据库中的agent类型"""
    # 这里需要实现更新数据库的逻辑
    # 暂时先返回True
    return True

# 以下函数保持不变（保持原有功能）
def get_chat_sessions(session_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        chat_messages: Optional[List[ChatMessageTable]] = session.query(ChatMessageTable) \
            .join(ChatSessionTable) \
            .filter(
            ChatSessionTable.session_id == session_id
        ).all()

        result = []
        if chat_messages:
            for record in chat_messages:
                result.append({
                    "id": record.id, "create_time": record.create_time,
                    "feedback": record.feedback, "feedback_time": record.feedback_time,
                    "role": record.role, "content": record.content
                })

        return result

def delete_chat_session(session_id: str) -> bool:
    with SessionLocal() as session:
        session_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if session_id is None:
            return False

        session.query(ChatMessageTable).where(ChatMessageTable.chat_id == session_id[0]).delete()
        session.query(ChatSessionTable).where(ChatSessionTable.id == session_id[0]).delete()
        session.commit()

    return True

def change_message_feedback(session_id: str, message_id: int, feedback: bool) -> bool:
    with SessionLocal() as session:
        id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if id is None:
            return False

        record = session.query(ChatMessageTable).filter(ChatMessageTable.id == message_id,
                                                        ChatMessageTable.chat_id == id[0]).first()
        if record is not None:
            record.feedback = feedback
            record.feedback_time = datetime.now()
            session.commit()

        return True

def list_chat(user_name: str) -> Optional[List[Any]]:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if user_id:
            chat_records: Optional[List[ChatSessionTable]] = session.query(
                                         ChatSessionTable.user_id,
                                         ChatSessionTable.session_id,
                                         ChatSessionTable.title,
                                         ChatSessionTable.start_time).filter(ChatSessionTable.user_id == user_id[0]).all()
            if chat_records:
                return [ChatSession(user_id = x.user_id, session_id=x.session_id, title=x.title, start_time=x.start_time) for x in chat_records]
            else:
                return []
        else:
            return []

def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        message_recod = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if message_recod:
            message_recod = ChatMessageTable(
                chat_id=message_recod[0],
                role=role,
                content=content
            )
            session.add(message_recod)
            session.commit()