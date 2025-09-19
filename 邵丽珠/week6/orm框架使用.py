from sqlalchemy import create_engine,Column,Integer,String
from sqlalchemy.orm import sessionmaker,declarative_base,relationship

engine = create_engine('sqlite:///alice.db',echo=True)

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    name = Column(String,nullable=False)
    age = Column(Integer)

    def __repr__(self):
        return '<User %r>' % self.name

Base.metadata.create_all(engine)
print("数据库和表已创建")

Session = sessionmaker(bind=engine)
session = Session()
Alice = User(name='爱丽丝',age=18)
Bob = User(name='小明',age=20)
Kipper = User(name='哈哈',age=10)

session.add_all([Alice,Bob,Kipper])
session.commit()
print("数据已成功插入")

result=session.query(User).all()
for user in result:
    print(f"姓名：{user.name},年龄：{user.age}")

update_user=session.query(User).filter_by(name="小明").first()
if update_user:
    update_user.age=40
    session.commit()
    print("小明年龄已更新")

update_user=session.query(User).filter_by(name="小明").first()
if update_user:
    print(f"{update_user.name}更新后的年龄是{update_user.age}")

delete_user = session.query(User).filter_by(name="哈哈").first()
if delete_user:
    session.delete(delete_user)
    session.commit()
    print(f"{delete_user.name}已删除")

users=session.query(User).all()
for user in users:
    print(f"姓名：{user.name},年龄：{user.age}")

session.close()
print("会话已关闭。")