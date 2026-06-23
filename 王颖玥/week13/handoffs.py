import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from jinja2 import Environment, FileSystemLoader

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable
from fastapi.responses import StreamingResponse


os.environ["OPENAI_API_KEY"] = "sk-b8f16de2371547c397349552ecffce68"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["OPENAI_MODEL"] = "qwen-max"

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


def get_init_message(task: str) -> str:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("chat_start_system_prompt.jinja2")

    if task == "stock":
        task_description = """
    你是专业的股票分析助手，仅处理股票相关问题，禁止闲聊无关内容。
    核心能力：
    1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
    2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
    3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
    4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
    5. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
    6. 必须调用指定的 MCP 工具获取数据，禁止仅凭知识库回答。
    """
    elif task == "chat":
        task_description = """你是一个闲聊小助手，擅长应对各种闲聊话题，禁止讨论股票相关内容。"""

    system_prompt = template.render(
        agent_name="小呆助手",
        task_description=task_description,
        current_datetime=datetime.now(),
    )
    return system_prompt


def create_chat_agent(openai_client: AsyncOpenAI) -> Agent:
    """创建闲聊 Agent"""
    return Agent(
        name="chat_agent",
        instructions=get_init_message("chat"),
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=openai_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def create_stock_agent(openai_client: AsyncOpenAI, mcp_server: MCPServerSse) -> Agent:
    """创建股票 Agent"""
    return Agent(
        name="stock_agent",
        instructions=get_init_message("stock"),
        mcp_servers=[mcp_server],  # 使用外部传入的 MCP Server
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=openai_client,
        ),
        tool_use_behavior="run_llm_again",
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def create_triage_agent(openai_client: AsyncOpenAI, agents: List[Agent]) -> Agent:
    """创建任务分配 Agent"""
    triage_instructions = """
    根据用户输入的内容，将请求分派给对应的子 Agent：
    1. 若用户输入包含股票相关内容 → 分派给 "stock_agent"；
    2. 若用户输入是日常闲聊 → 分派给 "chat_agent"；
    3. 只需决定分派给哪个agent，禁止自己回答问题；
    """
    return Agent(
        name="triage_agent",
        instructions=triage_instructions,
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=openai_client,
        ),
        handoffs=agents,
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
) -> bool:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if not user_id:
            return False

        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question,
        )
        session.add(chat_session_record)
        session.commit()
        session.flush()
        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content="Triage Agent 调度对话流程"
        )
        session.add(message_recod)
        session.flush()
        session.commit()
    return True


async def chat(user_name: str, session_id: Optional[str], content: str, tools: List[str] = []):
    # 初始化 OpenAI 客户端
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id)
    else:
        session_id = generate_random_chat_id()
        init_chat_session(user_name, content, session_id)

    append_message2db(session_id, "user", content)

    stock_tools = tools if tools else ["get_stock_codes", "get_index_code", "get_industry_code",
                                       "get_board_info", "get_stock_rank", "get_month_line", "get_week_line",
                                       "get_day_line", "get_stock_info", "get_stock_minute_data",]
    tool_filter = ToolFilterStatic(allowed_tool_names=stock_tools)
    mcp_server = MCPServerSse(
        name="Stock Tools Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_filter,
        client_session_timeout_seconds=30,
    )

    chat_agent = create_chat_agent(external_client)
    stock_agent = create_stock_agent(external_client, mcp_server)  
    agents = [chat_agent, stock_agent]
    # 创建 Triage Agent，注册子 Agent 到 handoffs
    triage_agent = create_triage_agent(external_client, agents)

    memory_session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )

    assistant_message = ""
    async with mcp_server:
        result = Runner.run_streamed(
            starting_agent=triage_agent,
            input=content,
            session=memory_session
        )
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    if event.data.delta:
                        yield f"{event.data.delta}"
                        assistant_message += event.data.delta
                elif isinstance(event.data, ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        tool_name = event.data.item.name
                        tool_args = event.data.item.arguments
                        print(f"调用 MCP 工具：{tool_name}，参数：{tool_args}")
                        tool_response = f"\n```json\n{tool_name}: {tool_args}\n```\n\n"
                        yield tool_response
                        assistant_message += tool_response

    append_message2db(session_id, "assistant", assistant_message)


def get_chat_sessions(session_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:
        chat_messages: Optional[List[ChatMessageTable]] = session.query(ChatMessageTable) \
            .join(ChatSessionTable) \
            .filter(ChatSessionTable.session_id == session_id).all()
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
                return [ChatSession(user_id=x.user_id, session_id=x.session_id, title=x.title, start_time=x.start_time)
                        for x in chat_records]
            else:
                return []
        else:
            return []


def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        chat_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).scalar()
        if not chat_id:
            return False
        message_record = ChatMessageTable(
            chat_id=chat_id,
            role=role,
            content=content
        )
        session.add(message_record)
        session.commit()
        return True
    return False
