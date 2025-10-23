import json
import pdfplumber
import time
import jwt
import requests
import numpy as np

import openai

client = openai.OpenAI(
    api_key="sk-9327b8540d1848c88ff106621f55a84a",
    base_url="https://api.deepseek.com"
)


def ask_deepseek(content):
    max_error = 0
    while max_error < 3:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": content}
                ],
                stream=False)
            return response.choices[0].message.content
        except Exception as e:
            max_error += 1

    return '无法回答'


bge = json.load(open('submit_bge_retrieval_top10.json'))
tfidf = json.load(open('submit_tfidf_retrieval_top10.json'))

questions = json.load(open('questions.json'))
pdf = pdfplumber.open('汽车知识手册.pdf')
pdf_content_dict = {}
for page_idx in range(len(pdf.pages)):
    pdf_content_dict['page_' + str(page_idx + 1)] = pdf.pages[page_idx].extract_text()

fusion_result = []
k = 60
for q1, q2 in zip(bge[:], tfidf[:]):
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
    q1['reference'] = sorted_dict[0][0]

    reference_pages = [int(x[0].split('_')[1]) for x in sorted_dict]
    reference_pages = np.array(reference_pages)

    reference_content = ''
    reference_content += pdf_content_dict[sorted_dict[0][0]].replace('\n', ' ') + "上述内容在第" + str(
        reference_pages[0] + 1) + "页"
    print(" question: ", q1['question'])
    print(" reference: ", pdf_content_dict[q1['reference']])

    reference_page = q1['reference'].split('_')[1]

    prompt = '''
    你是一个汽车专家，你擅长编写和回答汽车相关的用户提问，帮我结合给定的资料，回答下面的问题。
    1. 如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。
    2. 如果提问不符合逻辑，请回答无法回答。
    3. 如果问题可以从资料中获得，则请逐步回答。

    资料：{0}

    问题：{1}
    '''.format(reference_content, q1["question"])
    answer = ask_deepseek(prompt)
    if answer.startswith('无法回答'):
        answer = '结合给定的资料，无法回答问题'

    q1['answer'] = answer
    print(" answer: ", answer)
    print("\n\n\n")

    fusion_result.append(q1)

with open("submit_fusion_bge+tfidf_rerank_retrieval_deepseek.json", 'w', encoding='utf8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)
