import json
from datetime import datetime
from time import sleep
from typing import Union

from elasticsearch import Elasticsearch
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

# ORM
Base = declarative_base()


class Class(Base):
    __tablename__ = "t_class"

    class_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    grade = Column(Integer, nullable=False)

    # 关联学生
    students = relationship("Student", back_populates="cclass")

    def __repr__(self):
        return f"Class(name='{self.name}', grade={self.grade})"


class Student(Base):
    __tablename__ = "t_student"

    student_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)

    class_id = Column(Integer, ForeignKey("t_class.class_id"))

    cclass = relationship("Class", back_populates="students")

    def __repr__(self):
        return f"Student(name='{self.name}', age={self.age}, gender={self.gender})"


class ClassRecord(Base):
    __tablename__ = "t_class_record"

    record_id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("t_student.student_id"))
    class_id = Column(Integer, ForeignKey("t_class.class_id"))
    ops = Column(String, nullable=False)
    record_time = Column(DateTime, default=datetime.now, nullable=False)

    student = relationship("Student", lazy="joined")
    cclass = relationship("Class", lazy="joined")

    def __repr__(self):
        return f"ClassRecord(class={self.cclass.name},student={self.student.name}, ops='{self.ops}', record_time={self.record_time})"


def init_db(recreate: bool = False) -> Session:
    def generate_data_db(session, num_records: int = 100):
        from faker import Faker
        import random
        fake = Faker('zh_CN')
        # 插入班级数据
        class_records = []
        for i in range(9):
            for j in range(4):
                class_records.append(Class(name=f"{i + 1}年{j + 1}班", grade=i + 1))
        session.add_all(class_records)
        session.commit()

        print("初始数据已成功插入数据库")

        student_records = []
        for _ in range(num_records):
            gender = random.choice(['男', '女'])
            student = Student(
                name=fake.name_male() if gender == '男' else fake.name_female(),
                age=fake.random_int(min=6, max=15),
                gender=gender,
                cclass=random.choice(class_records))
            student_records.append(student)
        session.add_all(student_records)
        session.commit()
        return

    # 创建数据库引擎，这里使用 SQLite
    engine = create_engine("sqlite:///stu.db", echo=False)
    Session = sessionmaker(bind=engine)

    print("db连接成功！")

    session = Session()
    if recreate:
        # 清除所有表
        Base.metadata.drop_all(engine)
        # 创建数据库和表
        Base.metadata.create_all(engine)
        print("数据库和表已成功创建。")
        # 生成初始数据
        generate_data_db(session)
    return session


def init_es(recreate: bool = False):
    es = Elasticsearch("http://localhost:9200")
    if es.ping():
        print("es连接成功！")
    else:
        print("es连接失败。请检查 Elasticsearch 服务是否运行。")
        raise Exception("Elasticsearch connection failed")

    if recreate:
        il = es.indices.get(index="*")
        il = [i for i in il if not i.startswith(".")]
        for i in il:
            print(f"删除索引: {i}")
            es.indices.delete(index=i)

    return es


def ops_db(session: Session):
    def zhangsan_create():
        c1 = session.query(Class).first()
        # 插入学生数据
        zhangsan = Student(name="张三", age=6, gender="男", cclass=c1)
        lisi = Student(name="王五", age=7, gender="女", cclass=c1)
        session.add_all([zhangsan, lisi])
        session.commit()

        # 插入学籍记录
        record1 = ClassRecord(student=zhangsan, cclass=c1, ops="入学")
        record2 = ClassRecord(student=lisi, cclass=c1, ops="入学")
        session.add_all([record1, record2])
        session.commit()

    def zhangsan_check_in():
        zhangsan = session.query(Student).filter_by(name="张三").first()
        if not zhangsan:
            print("张三未入学")
            return
        print("张三在籍")
        return

        ############ 张三退学 ############

    def zhangsan_drop():
        zhangsan = session.query(Student).filter_by(name="张三").first()
        info = zhangsan.__dict__.copy()
        if not zhangsan:
            print("张三未入学")
            return None
        session.delete(zhangsan)
        print("张三已退学")
        session.commit()
        return info

    print("===== 张三入学 =====")
    zhangsan_create()

    print("===== 第一次查询张三入学记录 =====")
    zhangsan_check_in()

    print("===== 张三退学 =====")
    zhangsan = zhangsan_drop()
    print(zhangsan)

    print("===== 第二次查询张三入学记录 =====")
    zhangsan_check_in()


def etl_db2es(session: Session, es: Elasticsearch):
    index_name = "t_student"
    mapping = {
        "mappings": {
            "properties": {
                "student_id": {"type": "integer"},
                "name": {"type": "text", "analyzer": "ik_max_word"},
                "age": {"type": "integer"},
                "gender": {"type": "keyword"},
                "class_id": {"type": "integer"},
            }
        }
    }
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)

    # db -> es
    students = session.query(Student).all()
    for student in students:
        doc = {
            "student_id": student.student_id,
            "name": student.name,
            "age": student.age,
            "gender": student.gender,
            "class_id": student.class_id
        }
        es.index(index=index_name, id=str(student.student_id), document=doc)
    es.indices.refresh(index=index_name)

    # 全文检索，使用match查询
    query = {
        "query": {
            "multi_match": {
                "query": "飞",
                "fields": ["name"]
            }
        }
    }
    res = es.search(index=index_name, body=query)
    print_es_result(res)

    # 使用bool:{must, filter}精确过滤
    query = {
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {
                        "query": "芳",
                        "fields": ["name"]
                    }}
                ],
                "filter": [
                    {"term": {"gender": "女"}},
                    {"range": {"age": {"gte": 10}}}
                ]
            }
        }
    }
    res = es.search(index=index_name, body=query)
    print_es_result(res)

    # 按关键词分组聚合
    query = {
        "aggs": {
            "count_by_gender": {
                "terms": {
                    "field": "gender",
                    "size": 10
                }
            }
        },
        "size": 0
    }
    res = es.search(index=index_name, body=query)
    print(json.dumps(res["aggregations"]["count_by_gender"]["buckets"], ensure_ascii=False, indent=2))


def print_es_result(res):
    print(f"总命中数: {res['hits']['total']['value']}")
    for hit in res['hits']['hits']:
        print(f"ID: {hit['_id']}, Score: {hit['_score']}, Source: {hit['_source']}")


if __name__ == '__main__':
    RECREATE = False

    # ORM
    session = init_db(recreate=RECREATE)
    ops_db(session)

    # elasticsearch
    es = init_es(recreate=RECREATE)
    etl_db2es(session, es)

    session.close()
