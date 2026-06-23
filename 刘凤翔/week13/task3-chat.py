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

from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable


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


def get_chat_agent_instructions(agent_type: str, task: str = "") -> str:
    """获取不同agent的系统提示词"""
    env = Environment(loader=FileSystemLoader("templates"))
    
    if agent_type == "stock":
        template = env.get_template("stock_agent_system_prompt.jinjia2")
        return template.render(
            agent_name="股票分析助手",
            current_datetime=datetime.now(),
        )
    elif agent_type == "general":
        template = env.get_template("general_chat_system_prompt.jinjia2")
        return template.render(
            agent_name="小呆助手",
            current_datetime=datetime.now(),
        )
    else:
        # 路由agent
        template = env.get_template("router_agent_system_prompt.jinjia2")
        return template.render(
            current_datetime=datetime.now(),
        )


def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        task: str,
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

        # 为路由agent初始化系统消息
        message_record = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=get_chat_agent_instructions("router")
        )
        session.add(message_record)
        session.commit()

    return True


class ChatAgentManager:
    """Agent管理器，负责创建和管理不同类型的agent"""
    
    def __init__(self):
        self.external_client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )
    
    def create_stock_agent(self, session: AdvancedSQLiteSession) -> Agent:
        """创建股票分析agent"""
        stock_tools = [
            "get_stock_code", "get_index_code", "get_industry_code", 
            "get_board_info", "get_stock_rank", "get_month_line",
            "get_week_line", "get_day_line", "get_stock_info", 
            "get_stock_minute_data"
        ]
        
        mcp_server = MCPServerSse(
            name="Stock Tools Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            tool_filter=ToolFilterStatic(allowed_tool_names=stock_tools),
            client_session_timeout_seconds=20,
        )
        
        return Agent(
            name="StockAnalysisAgent",
            instructions=get_chat_agent_instructions("stock"),
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=self.external_client,
            ),
            tool_use_behavior="run_llm_again",
            model_settings=ModelSettings(parallel_tool_calls=False)
        )
    
    def create_general_agent(self, session: AdvancedSQLiteSession) -> Agent:
        """创建通用聊天agent"""
        general_tools = [
            "get_today_daily_news", "get_douyin_hot_news", "get_github_hot_news",
            "get_toutiao_hot_news", "get_sports_news", "get_city_weather",
            "get_address_detail", "get_tel_info", "get_scenic_info",
            "get_flower_info", "get_rate_transform", "get_today_familous_saying",
            "get_today_motivation_saying", "get_today_working_saying"
        ]
        
        mcp_server = MCPServerSse(
            name="General Tools Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            tool_filter=ToolFilterStatic(allowed_tool_names=general_tools),
            client_session_timeout_seconds=20,
        )
        
        return Agent(
            name="GeneralChatAgent",
            instructions=get_chat_agent_instructions("general"),
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=self.external_client,
            ),
            tool_use_behavior="run_llm_again",
            model_settings=ModelSettings(parallel_tool_calls=False)
        )
    
    def create_router_agent(self, session: AdvancedSQLiteSession) -> Agent:
        """创建路由agent，负责决定使用哪个专业agent"""
        return Agent(
            name="RouterAgent",
            instructions=get_chat_agent_instructions("router"),
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=self.external_client,
            ),
            handoffs=[],  # 将在运行时动态设置
            model_settings=ModelSettings(parallel_tool_calls=False)
        )


async def chat(
    user_name: str, 
    session_id: Optional[str], 
    task: Optional[str], 
    content: str, 
    tools: List[str] = []
) -> AsyncGenerator[str, None]:
    """主要的聊天处理函数，支持多agent协同"""
    
    # 初始化会话
    if not session_id:
        session_id = generate_random_chat_id()
        init_chat_session(user_name, content, session_id, task or "通用聊天")
    
    # 存储用户消息
    append_message2db(session_id, "user", content)
    
    # 创建session和agent管理器
    session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )
    
    agent_manager = ChatAgentManager()
    
    # 如果有明确的task参数，直接使用对应的agent
    if task == "股票分析":
        agent = agent_manager.create_stock_agent(session)
        async for chunk in run_agent_with_streaming(agent, content, session, session_id):
            yield chunk
    elif task == "数据BI":
        # 可以后续添加数据BI agent
        agent = agent_manager.create_general_agent(session)
        async for chunk in run_agent_with_streaming(agent, content, session, session_id):
            yield chunk
    else:
        # 使用路由agent进行智能分发
        async for chunk in run_router_agent(agent_manager, content, session, session_id):
            yield chunk


async def run_agent_with_streaming(
    agent: Agent, 
    content: str, 
    session: AdvancedSQLiteSession, 
    session_id: str
) -> AsyncGenerator[str, None]:
    """运行单个agent并处理流式输出"""
    result = Runner.run_streamed(agent, input=content, session=session)
    
    assistant_message = ""
    current_tool_name = ""
    need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
    
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # 处理工具调用
            if isinstance(event.data, ResponseOutputItemDoneEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    current_tool_name = event.data.item.name
                    tool_output = f"\n```json\n{event.data.item.name}: {event.data.item.arguments}\n```\n\n"
                    yield tool_output
                    assistant_message += tool_output
            
            # 处理文本输出
            elif isinstance(event.data, ResponseTextDeltaEvent):
                if event.data.delta:
                    yield event.data.delta
                    assistant_message += event.data.delta
    
    # 存储助手回复
    append_message2db(session_id, "assistant", assistant_message)


async def run_router_agent(
    agent_manager: ChatAgentManager,
    content: str,
    session: AdvancedSQLiteSession,
    session_id: str
) -> AsyncGenerator[str, None]:
    """运行路由agent进行智能分发"""
    
    # 创建路由agent（不带handoffs）
    router_agent = agent_manager.create_router_agent(session)
    
    # 分析用户意图
    intent_analysis_prompt = f"""
    请分析用户的问题意图，并返回对应的agent类型：
    
    用户问题: {content}
    
    可选的agent类型：
    - stock: 当问题涉及股票、股价、K线、大盘、行业板块、股票代码、股票排名、股票信息等金融投资相关内容时
    - general: 当问题涉及新闻、天气、工具查询、日常聊天、名言警句等通用内容时
    
    请只返回 "stock" 或 "general"，不要返回其他内容。
    """
    
    result = Runner.run(router_agent, input=intent_analysis_prompt, session=session)
    agent_type = result.final_output.strip().lower()
    
    # 根据分析结果选择合适的agent
    if agent_type == "stock":
        yield "🔍 检测到股票相关问题，正在调用股票分析专家...\n\n"
        agent = agent_manager.create_stock_agent(session)
    else:
        yield "💬 正在调用通用聊天助手...\n\n"
        agent = agent_manager.create_general_agent(session)
    
    # 运行选择的agent
    async for chunk in run_agent_with_streaming(agent, content, session, session_id):
        yield chunk


# 以下保持原有的辅助函数不变
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
        session_id_record = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if session_id_record is None:
            return False

        session.query(ChatMessageTable).where(ChatMessageTable.chat_id == session_id_record[0]).delete()
        session.query(ChatSessionTable).where(ChatSessionTable.id == session_id_record[0]).delete()
        session.commit()
    return True


def change_message_feedback(session_id: str, message_id: int, feedback: bool) -> bool:
    with SessionLocal() as session:
        id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if id is None:
            return False

        record = session.query(ChatMessageTable).filter(
            ChatMessageTable.id == message_id,
            ChatMessageTable.chat_id == id[0]
        ).first()
        if record is not None:
            record.feedback = feedback
            record.feedback_time = datetime.now()
            session.commit()
        return True


def list_chat(user_name: str) -> Optional[List[Any]]:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if user_id:
            chat_records = session.query(
                ChatSessionTable.user_id,
                ChatSessionTable.session_id,
                ChatSessionTable.title,
                ChatSessionTable.start_time
            ).filter(ChatSessionTable.user_id == user_id[0]).all()
            if chat_records:
                from models.data_models import ChatSession
                return [ChatSession(
                    user_id=x.user_id, 
                    session_id=x.session_id, 
                    title=x.title, 
                    start_time=x.start_time
                ) for x in chat_records]
        return []


def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        session_record = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if session_record:
            message_record = ChatMessageTable(
                chat_id=session_record[0],
                role=role,
                content=content
            )
            session.add(message_record)
            session.commit()
            return True
    return False