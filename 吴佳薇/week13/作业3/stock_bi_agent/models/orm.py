from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
class Base(DeclarativeBase):
    pass


class UserTable(Base):
    __tablename__ = 'user'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_name: Mapped[str] = mapped_column(String)
    user_role: Mapped[str] = mapped_column(String)
    password: Mapped[str] = mapped_column(String)
    register_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[bool] = mapped_column(Boolean)

class DataTable(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    data_type = Column(String)
    create_user_id = Column(Integer, ForeignKey('user.id'))
    create_time = Column(DateTime)
    alter_time = Column(DateTime)

class UserFavoriteStockTable(Base):
    __tablename__ = 'user_favorite_stock'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stock_id: Mapped[str] = mapped_column(String)
    user_id: Mapped[int] = Column(Integer, ForeignKey('user.id'))
    create_time: Mapped[datetime] = Column(DateTime, default=datetime.utcnow)

class ChatSessionTable(Base):
    """表1，存储一次对话列表的基础信息，聊天会话模型：存储用户会话的元数据。"""
    __tablename__ = 'chat_session'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    session_id = Column(String)
    title = Column(String(100))
    start_time = Column(DateTime, default=datetime.now)
    feedback: Mapped[bool] = mapped_column(Boolean, nullable=True)
    feedback_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    # 新增字段
    agent_type = Column(String, default="chat")  # 'chat' 或 'stock'

class ChatMessageTable(Base):
    """表2，存储一次对话的每一条记录，聊天消息模型：存储会话中的每一条消息。"""
    __tablename__ = 'chat_message'

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey('chat_session.id'), nullable=False)
    role = Column(String(10), nullable=True)
    content = Column(Text, nullable=True)
    generated_sql = Column(Text, nullable=True)
    generated_code = Column(Text, nullable=True)
    create_time = Column(DateTime, default=datetime.now)
    feedback: Mapped[bool] = mapped_column(Boolean, nullable=True)
    feedback_time: Mapped[datetime] = mapped_column(DateTime, nullable=True)

DATABASE_URL = "sqlite:///./assert/sever.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
