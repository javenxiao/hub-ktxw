import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INDEX_NAME = "tools"

INDEX_MAPPING = {
    'mappings': {
        'properties': {
            'tool': {
                'type': 'text'
            },
            'text': {
                'type': 'text'
            },
            'text_vector': {
                'type': 'dense_vector',
                'dims': 512,
                'index': True,
                'similarity': 'cosine'
            }
        }
    }
}

# 数据
datas = [
    {
        "tool": "student_performance",
        "text": "在教育培训领域，评估学生的学习效果是衡量教学质量和课程成效的重要环节。为了更系统\n地理解和预测学生在课程中的表现，构建了一个基于关键影响因素的确定性模型。该模型综\n合考虑了学习时长、出勤率、平时测验成绩以及课堂参与度四个核心变量，旨在通过量化方\n式反映学生的学习成果，并模拟其在学习过程中的非线性增长趋势。\n该模型可用于学生表现预测、教学反馈分析以及个性化学习路径优化等场景，为教育决策提\n供数据支持。"
    },
    {
        "tool": "dissolved_oxygen",
        "text": "在水产养殖系统中，溶解氧（Dissolved Oxygen, DO）是影响水生生\n物健康和生长的关键环境因子之一。其浓度受多种因素影响，包括水体自净能力、微生物活\n动、水生生物呼吸作用以及外界环境（如温度、光照、风力等）的周期性变化。为了更好地\n理解和预测DO的动态变化趋势，建立一个能够反映其非线性行为的数学模型具有重要意义\n。该模型可用于模拟封闭或半封闭养殖系统中DO浓度随时间演变的过程，为水质调控和管\n理提供理论支持。"
    },
    {
        "tool": "tun",
        "text": "在复杂系统分析中，常常需要构建能够反映变量间非线性交互作用的数学模型。该模型旨在\n模拟两个输入变量x 和y 对某一目标输出的综合影响，其中包含了周期性\n变化与线性交互的成分。该建模方法适用于描述如环境因素对系统响应的影响、多因子耦合\n作用下的信号响应机制等场景。尽管模型本身为确定性函数，但其结构设计使得输出呈现出\n类随机波动的特性，从而更好地模拟真实世界中的复杂行为。"
    },
    {
        "tool": "predict_daily_weight_gain",
        "text": "在畜牧业生产中，饲料摄入量是影响牲畜生长性能的关键因素之一。为了量化饲料摄入量对\n牲畜日增重的影响，建立一个简化的代数模型，有助于理解在不同饲喂水平下牲畜的增重响\n应规律。该模型综合考虑了基础增重、饲料促进效应以及随着摄入量增加而出现的边际效益\n递减现象，适用于初步评估饲料利用效率和优化饲喂策略。"
    },
    {
        "tool": "logistic_growth_model",
        "text": "在畜牧业管理中，理解与预测牛群数量的动态变化具有重要意义。为了反映种群在有限资源\n环境下的增长特性，采用逻辑斯蒂增长模型的思想，构建一个一阶非线性差分方程模型。该\n模型不仅考虑了牛群的自然增长率，还引入了环境承载能力的限制因素，从而更真实地反映\n实际种群增长过程。该模型适用于对中长期牛群数量趋势进行预测，辅助制定合理的养殖策\n略与资源分配方案。"
    },
    {
        "tool": "loan_balance_update",
        "text": "在金融服务业中，贷款管理是核心业务之一。为了有效跟踪客户的贷款余额变化，金融机构\n需要建立数学模型来描述贷款本金随时间的动态演变。该模型可用于风险控制、信用评估以\n及贷款回收预测等场景。\n本模型聚焦于一个简化的贷款余额更新过程，旨在通过差分方程的形式刻画客户每月偿还固\n定金额后，贷款余额的变化情况。模型假设不考虑利息、违约风险及其他复杂因素，仅关注\n本金的递减过程，适用于初步理解贷款动态或作为更复杂模型的基础模块。"
    },
    {
        "tool": "calculate_limit_bending_moment",
        "text": "在建筑工程中，钢筋混凝土梁的抗弯承载力是结构设计和安全性评估的重要指标。为确保结\n构体系在设计使用寿命内的可靠性和稳定性，需要对梁在弯矩作用下的承载能力进行合理估\n算。该建模旨在通过引入混凝土抗压强度、纵向受拉钢筋面积以及梁的有效高度等关键参数\n，构建一个非线性关系来模拟梁的极限弯矩承载能力。该模型可用于初步设计阶段的快速评\n估，也可作为教学和研究中的参考工具。"
    },
    {
        "tool": "model_with_noise",
        "text": "在实际系统建模中，许多物理过程或观测数据往往受到不可控因素的影响，例如测量误差、\n环境噪声或其他随机扰动。为了更贴近现实场景，建模过程中通常引入随机性成分，以反映\n系统的不确定性和复杂性。本模型旨在模拟一个带有随机噪声的确定性函数，其输出不仅依\n赖于输入变量的数值关系，还受到随机扰动的影响，从而更真实地反映实际问题中的波动特\n性。"
    },
    {
        "tool": "linear_model",
        "text": "在构建定量分析模型时，线性关系是最基础且广泛应用的数学表达形式。该模型假设输出变\n量与输入变量之间存在线性相关性，即输出变量随输入变量按固定比例变化，并包含一个固\n定的偏移量。此类模型广泛应用于回归分析、系统建模以及机器学习的初期理论构建中。通\n过设定权重和偏置项，可以对输入数据进行线性变换，以预测输出结果。\n本模型旨在模拟一个单变量线性函数，用于描述输入值经过线性运算后所对应的输出值，适\n用于基础教学、算法验证以及模型初始化等场景。"
    },
    {
        "tool": "calculate_adg",
        "text": "在水产养殖中，鱼类的平均日增重（ADG, Average Daily Gain）\n是衡量养殖效益和生长性能的重要指标。为了科学地预测和优化鱼类生长情况，建立一个基\n于关键环境与管理因子的预测模型具有重要意义。该模型综合考虑了水温、溶解氧、pH值\n、饲料投喂率以及养殖密度五个对鱼类生长具有显著影响的因素，旨在为养殖决策提供数据\n支持与理论依据。\n模型构建过程中，假设各因子在合理范围内对鱼类生长存在线性影响关系。其中，水温、溶\n解氧、pH值和投喂率被认为对鱼类生长具有正向促进作用，而养殖密度则因可能引发竞争\n压力和环境恶化而对生长产生抑制作用。"
    }
]

# 地址
hosts = [{
    'host': 'localhost',
    'port': 9200,
    'scheme': 'http'
}]

# 认证信息
auth = ('elastic', 'qhNdREadcp6ECGacW6Iw')
# 创建ES
es = Elasticsearch(hosts=hosts, basic_auth=auth)

# 测试连接
if not es.ping():
    print('Elasticsearch连接失败')
else:
    print('Elasticsearch连接成功')

# 加载模型
embedding_model = SentenceTransformer('../models/BAAI/bge-small-zh-v1.5')
embedding_model.eval()
rerank_tokenizer = AutoTokenizer.from_pretrained('../models/bge-reranker-base')
rerank_model = AutoModelForSequenceClassification.from_pretrained('../models/bge-reranker-base')
rerank_model.eval()


def build():
    """
    构建 RAG
    :return: 无
    """
    # 删除索引
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print('Elasticsearch删除索引成功 -> ', INDEX_NAME)
    else:
        print('Elasticsearch索引不存在 -> ', INDEX_NAME)
    # 创建索引
    if not es.indices.exists(index=INDEX_NAME):
        try:
            es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
            print('Elasticsearch创建索引成功 -> ', INDEX_NAME)
        except Exception as e:
            print('Elasticsearch创建索引失败 -> ', INDEX_NAME, " - ", e)
    else:
        print('Elasticsearch索引已存在 -> ', INDEX_NAME)
    # 插入文档
    documents = [
        {
            'tool': data['tool'],
            'text': data['text'],
            'text_vector': embedding_model.encode(data['text']).tolist()
        }
        for data in datas
    ]
    # 插入文档
    for document in documents:
        try:
            es.index(index=INDEX_NAME, body=document)
            print('插入文档成功 -> ', INDEX_NAME, " - ", document)
        except:
            print('插入文档失败 -> ', INDEX_NAME, " - ", document)


def search(query):
    """
    搜索工具
    :param query: 提问内容
    :return: 返回 top3 工具
    """
    query_vector = embedding_model.encode(query)
    # Embedding检索
    vector_items = list()
    vector_response = es.search(
        index=INDEX_NAME,
        body={
            "knn": {
                "field": "text_vector",
                "query_vector": query_vector,
                "k": 5,  # 返回前5结果
                "num_candidates": 10
            },
            "fields": ["tool", "text"],  # 返回 tool, text 字段
            "_source": False  # 不返回整个文档源
        }
    )
    for hit in vector_response['hits']['hits']:
        score = hit['_score']
        tool = hit['fields']['tool'][0]
        text = hit['fields']['text'][0]
        print(f"embedding检索 得分: {score:.4f}\n工具：{tool}\n内容: {text}")
        vector_items.append({
            "score": score,
            "tool": tool,
            "text": text
        })
    # 普通文本检索
    text_items = list()
    text_response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": 5,  # 返回前5结果
            "fields": ["tool", "text"],  # 返回 tool, text 字段
            "_source": False  # 不返回整个文档源
        })
    for hit in text_response['hits']['hits']:
        score = hit['_score']
        tool = hit['fields']['tool'][0]
        text = hit['fields']['text'][0]
        print(f"文本检索 得分: {score:.4f}\n工具：{tool}\n内容: {text}")
        text_items.append({
            "score": score,
            "tool": tool,
            "text": text
        })
    # 文本-工具 映射表
    text_tool_map = dict()
    # 合并结果并排序
    fusion_score = dict()
    k = 20
    for index, item in enumerate(vector_items):
        text = item['text']
        tool = item['tool']
        text_tool_map[text] = tool
        if text in fusion_score:
            fusion_score[text] += 1 / (k + index)
        else:
            fusion_score[text] = 1 / (k + index)
    for index, item in enumerate(text_items):
        text = item['text']
        tool = item['tool']
        text_tool_map[text] = tool
        if text in fusion_score:
            fusion_score[text] += 1 / (k + index)
        else:
            fusion_score[text] = 1 / (k + index)
    print("排序前：")
    for score in fusion_score.items():
        print(score)
    sort_score = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
    print("排序后：")
    for score in sort_score:
        print(score)
    # rerank重排序
    pairs = []
    for score in sort_score[:5]:
        pairs.append([query, score[0]])
    print(pairs)
    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        normalized_scores = torch.sigmoid(scores).tolist()
        print(normalized_scores)
    rerank_results = list()
    for index, score in enumerate(sort_score[:5]):
        rerank_result = {
                "tool": text_tool_map[score[0]],
                "text": score[0],
                "rerank_score": normalized_scores[index]
            }
        rerank_results.append(rerank_result)
        print(rerank_result)
    rerank_sort_result = sorted(rerank_results, key=lambda item: item["rerank_score"], reverse=True)
    return rerank_sort_result[:3]



if __name__ == '__main__':
    build()
    # query = "在畜牧业管理中，针对初始体重为200kg、生长速率为0.01/天的牛群，经过50天后的体重应为多少？"
    # search(query)
