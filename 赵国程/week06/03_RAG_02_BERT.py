import json
import pdfplumber
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('../model/BAAI/bge-small-zh-v1.5')

questions = json.load(open('questions.json'))
pdf = pdfplumber.open('汽车知识手册.pdf')
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })


question_sentences = [x['question'] for x in questions]
pdf_content_sentences = [x['content'] for x in pdf_content]

question_embeddings = model.encode(question_sentences, normalize_embeddings=True)
pdf_embeddings = model.encode(pdf_content_sentences, normalize_embeddings=True)

# 计算所有问题与所有PDF内容的相似度
scores = question_embeddings @ pdf_embeddings.T

# 选出每个问题相似度最高的PDF内容的索引
top1_indices = scores.argmax(axis=1)

# 选出每个问题相似度最高的前十个PDF内容的索引
top10_indices = scores.argsort(axis=1)[:, ::-1][:, :10]

#
for query_idx, feat in enumerate(question_embeddings):
    score = feat @ pdf_embeddings.T
    max_score_page_idx = score.argsort()[::-1][0] + 1
    questions[query_idx]['reference'] = ['page_' + str(max_score_page_idx)]

with open('submit_bge_retrieval_top1.json', 'w', encoding='utf-8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)

for query_idx, feat in enumerate(question_embeddings):
    score = feat @ pdf_embeddings.T
    max_score_page_index = score.argsort()[::-1] + 1
    questions[query_idx]['reference'] = ['page_' + str(x) for x in max_score_page_index[:10]]

with open('submit_bge_retrieval_top10.json', 'w', encoding='utf-8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)


