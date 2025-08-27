# SVM特点（效率较慢）：
#   1、在高维空间中寻找最优分离超平面
#   2、对特征缩放敏感，TF-IDF比纯词频更适合
#   3、强大的分类能力但计算复杂度较高

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # 使用TF-IDF效果更好
from sklearn.svm import SVC  # 导入支持向量机

# 数据加载
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 文本预处理
input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 特征提取
vector = TfidfVectorizer()
vector.fit(input_sentences.values)
input_feature = vector.transform(input_sentences.values)

# 使用SVM模型
svm_model = SVC(kernel='linear', probability=True)  # 线性核函数，启用概率预测
svm_model.fit(input_feature, dataset[1].values)  # 训练模型

# 预测新文本
test_query = "帮我完成这个任务"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

print("待预测的文本:", test_query)
print("SVM预测结果:", svm_model.predict(test_feature))
print("预测概率:", svm_model.predict_proba(test_feature))