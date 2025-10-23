import sqlite3
import json
from datetime import datetime

# 创建SQLite数据库连接
conn = sqlite3.connect('experiment_data.db')
cursor = conn.cursor()

# 创建数据表
cursor.execute('''
CREATE TABLE IF NOT EXISTS research_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT,
    publish_date DATE,
    category TEXT,
    citations INTEGER DEFAULT 0,
    keywords TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# 创建索引以提高查询性能
cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON research_papers(category)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations ON research_papers(citations)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_publish_date ON research_papers(publish_date)')

conn.commit()
print("SQLite数据库和表创建成功")

# 示例数据 - 研究论文信息
papers = [
    {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
        "publish_date": "2017-06-12",
        "category": "NLP",
        "citations": 78500,
        "keywords": "transformer, attention, neural machine translation"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "publish_date": "2018-10-11",
        "category": "NLP",
        "citations": 48500,
        "keywords": "bert, pre-training, language representation"
    },
    {
        "title": "Residual Learning for Image Recognition",
        "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
        "publish_date": "2015-12-10",
        "category": "Computer Vision",
        "citations": 96500,
        "keywords": "resnet, deep learning, image recognition"
    },
    {
        "title": "Generative Adversarial Networks",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
        "authors": "Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio",
        "publish_date": "2014-06-10",
        "category": "Generative Models",
        "citations": 68500,
        "keywords": "gan, generative models, adversarial training"
    },
    {
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "abstract": "There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently.",
        "authors": "Olaf Ronneberger, Philipp Fischer, Thomas Brox",
        "publish_date": "2015-05-18",
        "category": "Computer Vision",
        "citations": 38500,
        "keywords": "u-net, image segmentation, biomedical"
    }
]

# 插入数据
for paper in papers:
    cursor.execute('''
    INSERT INTO research_papers (title, abstract, authors, publish_date, category, citations, keywords)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        paper['title'],
        paper['abstract'],
        paper['authors'],
        paper['publish_date'],
        paper['category'],
        paper['citations'],
        paper['keywords']
    ))

conn.commit()
print(f"成功插入 {len(papers)} 条数据到SQLite")

# 查询所有数据
print("\n=== SQLite中的所有研究论文 ===")
cursor.execute("SELECT id, title, category, citations FROM research_papers")
all_papers = cursor.fetchall()
for paper in all_papers:
    print(paper)

# 按类别查询
print("\n=== NLP类别的论文 ===")
cursor.execute("SELECT id, title, citations FROM research_papers WHERE category = ?", ("NLP",))
nlp_papers = cursor.fetchall()
for paper in nlp_papers:
    print(paper)

# 按引用量排序查询
print("\n=== 引用量最高的3篇论文 ===")
cursor.execute("SELECT title, citations FROM research_papers ORDER BY citations DESC LIMIT 3")
top_cited = cursor.fetchall()
for paper in top_cited:
    print(paper)

# 关闭连接
conn.close()