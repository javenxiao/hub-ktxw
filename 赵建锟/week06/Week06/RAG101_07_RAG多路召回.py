import json
bge = json.load(open('submit_bge_sgement_retrieval_top10.json',encoding='UTF-8'))
bm25 = json.load(open('submit_bm25_retrieval_top10.json',encoding='UTF-8'))

fusion_result = []
k = 60
for q1, q2 in zip(bge, bm25):
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
    fusion_result.append(q1)

with open('submit_fusion_bge+bm25_retrieval.json', 'w', encoding='utf8') as up:
    json.dump(fusion_result, up, ensure_ascii=False, indent=4)