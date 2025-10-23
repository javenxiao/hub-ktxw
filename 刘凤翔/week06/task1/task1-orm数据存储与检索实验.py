from sqlalchemy import create_engine, Column, Integer, String, Text, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# 创建基类
Base = declarative_base()


# 定义研究论文模型
class ResearchPaper(Base):
    __tablename__ = 'research_papers_orm'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    abstract = Column(Text, nullable=False)
    authors = Column(Text)
    publish_date = Column(Date)
    category = Column(String(50))
    citations = Column(Integer, default=0)
    keywords = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<ResearchPaper(id={self.id}, title='{self.title}', category='{self.category}')>"


# 创建数据库引擎和会话
engine = create_engine('sqlite:///experiment_data_orm.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# 创建示例数据
orm_papers = [
    ResearchPaper(
        title="Deep Residual Learning for Image Recognition",
        abstract="Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
        authors="Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
        publish_date=datetime.strptime("2015-12-10", "%Y-%m-%d").date(),
        category="Computer Vision",
        citations=96500,
        keywords="resnet, deep learning, image recognition"
    ),
    ResearchPaper(
        title="Generative Adversarial Networks",
        abstract="We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
        authors="Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio",
        publish_date=datetime.strptime("2014-06-10", "%Y-%m-%d").date(),
        category="Generative Models",
        citations=68500,
        keywords="gan, generative models, adversarial training"
    )
]

# 批量插入数据
session.bulk_save_objects(orm_papers)
session.commit()
print(f"通过ORM成功插入 {len(orm_papers)} 条数据")

# 查询所有文档
print("=== ORM中的所有研究论文 ===")
all_papers = session.query(ResearchPaper).all()
for paper in all_papers:
    print(paper)

# 条件查询
print("\n=== 计算机视觉类别的论文 ===")
cv_papers = session.query(ResearchPaper).filter(ResearchPaper.category == "Computer Vision").all()
for paper in cv_papers:
    print(paper)

# 复杂查询 - 引用量大于70000的论文
print("\n=== 引用量大于70000的论文 ===")
high_cited = session.query(ResearchPaper).filter(ResearchPaper.citations > 70000).all()
for paper in high_cited:
    print(f"{paper.title} - {paper.citations} citations")

# 更新数据
print("\n=== 更新论文引用量 ===")
paper_to_update = session.query(ResearchPaper).filter(ResearchPaper.title.like("%Generative Adversarial%")).first()
if paper_to_update:
    paper_to_update.citations += 1000
    session.commit()
    print(f"更新成功: {paper_to_update.title} 现在有 {paper_to_update.citations} 次引用")

# 删除数据
print("\n=== 删除特定论文 ===")
paper_to_delete = session.query(ResearchPaper).filter(ResearchPaper.title.like("%U-Net%")).first()
if paper_to_delete:
    session.delete(paper_to_delete)
    session.commit()
    print(f"删除成功: {paper_to_delete.title}")

# 关闭会话
session.close()