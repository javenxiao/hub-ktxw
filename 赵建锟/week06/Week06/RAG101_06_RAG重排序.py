import json
import pdfplumber
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 读取数据集
questions = json.load(open("questions.json",encoding='UTF-8'))
pdf = pdfplumber.open("汽车知识手册.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    pdf_content.append({
        'page': 'page_' + str(page_idx + 1),
        'content': pdf.pages[page_idx].extract_text()
    })

# 加载排序模型
tokenizer = AutoTokenizer.from_pretrained('../models/BAAI/bge-reranker-base/')
rerank_model = AutoModelForSequenceClassification.from_pretrained('../models/BAAI/bge-reranker-base/')
# rerank_model.cuda()
rerank_model.eval()

# 进行召回合并
bge = json.load(open('submit_bge_sgement_retrieval_top10.json',encoding='UTF-8'))
bm25 = json.load(open('submit_bm25_retrieval_top10.json',encoding='UTF-8'))

fusion_result = []
k = 60
for q1, q2 in zip(bge, bm25):
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

    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)

    pairs = []
    for sorted_result in sorted_dict[:3]:
        page_index = int(sorted_result[0].split('_')[1]) - 1
        pairs.append([q1["question"], pdf_content[page_index]['content']])

    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key] for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    sorted_result = sorted_dict[scores.cpu().numpy().argmax()]
    q1['reference'] = sorted_result[0]
    
    fusion_result.append(q1)

with open('submit_fusion_bge+bm25_rerank_retrieval.json', 'w', encoding='utf8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)