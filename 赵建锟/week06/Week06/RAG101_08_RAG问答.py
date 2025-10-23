import json
import pdfplumber
import time
import jwt
import requests
import numpy as np

# 实际KEY，过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

def ask_glm(content):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token("89e209c17a582a029626f784c1fd0777.gk44xUlgeaqoS610", 1000)
    }

    data = {
        "model": "glm-3-turbo",
        "messages": [{"role": "user", "content": content}]
    }

    response = requests.post(url, headers=headers, json=data)
    print("【大模型返回】")
    print(response)
    return response.json()

def ask_gpt(content):
    url = "https://openai.api2d.net/v1/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': "Bearer fk222903-VSeeeE4nL4I7MUbcuJEGpUEkVH6VgEnP"
    }

    max_error = 0
    while True:
        if max_error > 5:
            return "无法回答"
        try:
            data = {
                "model": "gpt-3.5-turbo-0613",
                "messages": [{"role": "user", "content": content}]
            }
        
            response = requests.post(url, headers=headers, json=data, timeout=10)
            # print("【大模型返回】\n" + response)
            return response.json()
        except:
            max_error += 1
            continue

bge = json.load(open('submit_bge_sgement_retrieval_top10.json',encoding='UTF-8'))
bm25 = json.load(open('submit_bm25_retrieval_top10.json',encoding='UTF-8'))

questions = json.load(open("questions.json",encoding='UTF-8'))
pdf = pdfplumber.open("汽车知识手册.pdf")
pdf_content_dict = {}
for page_idx in range(len(pdf.pages)):
    pdf_content_dict['page_' + str(page_idx + 1)] = pdf.pages[page_idx].extract_text()

fusion_result = []
k = 60
for q1, q2 in zip(bge[:], bm25[:]):
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
    q1['reference'] = sorted_dict[0][0]

    reference_pages = [int(x[0].split('_')[1]) for x in sorted_dict]
    reference_pages = np.array(reference_pages)

    reference_content = ''
    reference_content += pdf_content_dict[sorted_dict[0][0]].replace('\n', ' ') + '\t' + "上述内容在第" + str(reference_pages[0]) + '页'
    
    # if np.abs(reference_pages[0] - reference_pages[1]) <= 1:
    #     if reference_pages[0] < reference_pages[1]:
    #         reference_content += pdf_content_dict[sorted_dict[0][0]].replace('\n', ' ') + '\t' + "内容在第" + str(reference_pages[0]) + '页'
    #         reference_content + '\n'
    #         reference_content += pdf_content_dict[sorted_dict[1][0]].replace('\n', ' ') + '\t' + "内容在第" + str(reference_pages[1]) + '页'
    #     else:
    #         reference_content += pdf_content_dict[sorted_dict[1][0]].replace('\n', ' ') + '\t' + "内容在第" + str(reference_pages[1]) + '页'
    #         reference_content + '\n'
    #         reference_content += pdf_content_dict[sorted_dict[0][0]].replace('\n', ' ') + '\t' + "内容在第" + str(reference_pages[0]) + '页'
    # else:
    #     reference_content += pdf_content_dict[sorted_dict[0][0]].replace('\n', ' ') + '\t' + "内容在第" + str(reference_pages[0]) + '页'
        
    print("【用户提问】\n" + q1["question"])
    print("【参考资料】\n" + pdf_content_dict[q1['reference']])

    reference_page = q1['reference'].split('_')[1]
    
    prompt = '''你是一个汽车专家，你擅长编写和回答汽车相关的用户提问，帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。

资料：{0}


问题：{1}
    '''.format(reference_content, q1["question"])
    # answer = ask_glm(prompt)['choices'][0]['message']['content']

    answer = '无法'
    for _ in range(5):
        try:
            answer = ask_glm(prompt)['choices'][0]['message']['content']
            if answer:
                break
        except:
            continue
    
    if '无法' in answer:
        answer = '结合给定的资料，无法回答问题。'
    
    q1['answer'] = answer
    print("【模型回答】\n" + q1['answer'])

    print("\n\n\n")

    fusion_result.append(q1)

with open('submit_fusion_bge+bm25_rerank_retrieval_glm4.json', 'w', encoding='utf8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)