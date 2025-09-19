import json
import traceback
from datetime import datetime

from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, Boolean, or_

from my_data import DATAS

# 模型基类
Base = declarative_base()


# 文档类
class Document(Base):

    # 表名
    __tablename__ = 'document'

    # 字段
    doc_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    category = Column(String)
    author = Column(String)
    publish_date = Column(Date)
    views = Column(Integer)
    tags = Column(Text)
    is_recommended = Column(Boolean)

    @property
    def tags_list(self):
        if self.tags:
            return json.loads(self.tags)
        return []

    @tags_list.setter
    def tags_list(self, value):
        self.tags = json.dumps(value)

    @property
    def publish_date_text(self):
        if self.publish_date:
            return self.publish_date.strftime('%Y-%m-%d')
        return ''

    @publish_date_text.setter
    def publish_date_text(self, value):
        self.publish_date = datetime.strptime(value, '%Y-%m-%d').date()

    def __repr__(self):
        return f'<Document {self.doc_id}, {self.title}, {self.content}, {self.category}, {self.author}, {self.publish_date}, {self.views}, {self.tags}>'

# 创建引擎
engine = create_engine('sqlite:///data.db', echo=False)

# 创建数据库和数据表
Base.metadata.create_all(engine)
print("创建数据库和数据表成功")

# 创建会话
Session = sessionmaker(bind=engine)
session = Session()

# 新增数据
try:
    for item in DATAS:
        item['tags_list'] = item['tags']
        del item['tags']
        item['publish_date_text'] = item['publish_date']
        del item['publish_date']
        document = Document(**item)
        result = session.query(Document).filter_by(doc_id=document.doc_id).first()
        if not result:
            session.add(document)
            print('插入数据成功 -> ', document.doc_id)
        else:
            print('已经存在数据 -> ', result.doc_id)
    session.commit()
except Exception as e:
    print('插入数据失败')
    traceback.format_exc()
    session.rollback()

# 查询
results = session.query(Document) \
    .filter(or_(Document.title.like('%ElasticSearch%'), Document.content.like('%ElasticSearch%'))) \
    .all()
print('查找标题和内容有ElasticSearch内容的结果：')
for result in results:
    print(result)

# 更新
result = session.query(Document).filter_by(doc_id='DOC2024090020').first()
try:
    if result:
        print(f'更新前文档查看数 ->', result.views)
        result.views = 99999
        session.commit()
        print('更新文档成功')
    else:
        print('查找文档失败')
except Exception as e:
    print('更新文档失败')
    traceback.format_exc()
result = session.query(Document).filter_by(doc_id='DOC2024090020').first()
if result:
    print(f'更新后文档查看数 ->', result.views)

# 删除
result = session.query(Document).filter_by(doc_id='DOC2024090020').first()
try:
    if result:
        session.delete(result)
        session.commit()
        print('删除文档成功')
    else:
        print('查找文档失败')
except Exception as e:
    print('删除文档失败')
    traceback.format_exc()