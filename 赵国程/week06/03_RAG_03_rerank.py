import json
import pdfplumber
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 读取数据集
questions = json.load(open("questions.json"))
pdf = pdfplumber.open("汽车知识手册.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })

tokenizer = AutoTokenizer.from_pretrained('../model/BAAI/bge-reranker-base/')
rerank_model = AutoModelForSequenceClassification.from_pretrained('../model/BAAI/bge-reranker-base/')
rerank_model.eval()
bge = json.load(open('submit_bge_retrieval_top10.json'))
tfidf = json.load(open('submit_tfidf_retrieval_top10.json'))

fusion_result = []
k = 60
for q1, q2 in zip(bge, tfidf):
    print(len(fusion_result), len(bge))
    fusion_score = {}
    for idx, q in enumerate(q1['reference']):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    for idx, q in enumerate(q2['reference']):
        if q not in fusion_score:
            fusion_score[q] = 1 / (idx + k)
        else:
            fusion_score[q] += 1 / (idx + k)

    sorted_dict = sorted(fusion_score.items(), key=lambda x: x[1], reverse=True)

    pairs = []
    for sorted_result in sorted_dict[:3]:
        page_index = int(sorted_result[0].split('_')[1]) - 1
        pairs.append([q1['question'], pdf_content[page_index]['content']])

    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        inputs = {key: inputs[key] for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    sorted_result = sorted_dict[scores.cpu().numpy().argmax()]
    q1['reference'] = sorted_result[0]

    fusion_result.append(q1)

with open('submit_fusion_bge+bm25_rerank_retrieval.json', 'w', encoding='utf8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)
