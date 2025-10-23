from sqlalchemy import create_engine, Column, Integer, BigInteger, String
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# 使用mysql 数据库
engine = create_engine('mysql+pymysql://root:zhao19971107A@localhost:3306/test', echo=True)

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'

    id = Column(BigInteger, primary_key=True)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)
    phone_number = Column(String, nullable=True)

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()

test_user = User(username='test', password='123456')

session.add(test_user)
session.commit()

users = session.query(User).all()
for user in users:
    print(f"query: user.id: {user.id}, user.username: {user.username}, user.password: {user.password}, user.phone_number: {user.phone_number}")

print("update user")
user_to_update = session.query(User).filter_by(id=test_user.id).first()
if user_to_update:
    user_to_update.username = 'test_update'
    session.commit()

updated = session.query(User).filter_by(id=test_user.id).first()
if updated:
    print(f"updated: user.id: {updated.id}, user.username: {updated.username}, user.password: {updated.password}, user.phone_number: {updated.phone_number}")

user_to_delete = session.query(User).filter_by(id=test_user.id).first()
if user_to_delete:
    session.delete(user_to_delete)
    session.commit()
    print("delete user")

users = session.query(User).all()
for user in users:
    print(f"after delete: user.id: {user.id}, user.username: {user.username}, user.password: {user.password}, user.phone_number: {user.phone_number}")

