from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

engine = create_engine("sqlite:///mydatabase.db",echo = True)
Base = declarative_base()

class Student(Base):
    __tablename__='students'
    student_id = Column(Integer,primary_key=True)
    name = Column(String,nullable=False)
    address = Column(String)
    subject = relationship('Subject',back_populates="student")
    def __repr__(self):
        return f"<Student(name={self.name},address={self.address})>"

class Class(Base):
    __tablename__ = 'classes'
    address = Column(String,nullable=False)
    class_id = Column(Integer,primary_key = True)
    name = Column(String)
    def __repr__(self):
        return f"<Class(address={self.address},name={self.name})>"

class Subject(Base):
    __tablename__ = 'subjects'
    subject_id = Column(Integer,primary_key = True)
    name = Column(String)
    student = relationship('Student',back_populates='subject')
    student_id = Column(Integer,ForeignKey("students.student_id"))
    teacher = Column(String)
    def __repr__(self):
        return f"<Subject(name={self.name},teacher={self.teacher})>"
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
print("数据库和表已成功创建")

session = sessionmaker(bind = engine)
session = session()

xiaoming = Student(name = 'xiaoming',address = '湖南')
lihua = Student(name = '李华',address='上海')
session.add_all([lihua,xiaoming])

jisuanji = Subject(name='计算机',teacher = 'Bob',student = xiaoming)
gaodengshuxue = Subject(name = '高等数学',teacher= 'John',student = lihua)
session.add_all([jisuanji,gaodengshuxue])

session.commit()
print("数据已成功插入。")

print("\n--- 所有课程和它们的学生 ---")
# ORM 方式的 JOIN 查询
# 我们可以直接通过对象的属性来查询关联数据
results = session.query(Subject).join(Student).all()
for subject in results:
    print(f"课程: {subject.name}, 学生: {subject.student.name}")

update_state = session.query(Student).filter_by(name='xiaoming').first()
if update_state:
    update_state.address="北京"
    session.commit()

delete_state = session.query(Student).filter_by(name='李华').first()
if delete_state:
    session.delete(delete_state)
    session.commit()
print("剩余的学生")
query_state = session.query(Student).all()
for student in query_state:
    print(student.name,student.address)

session.close()
print("断开连接")
