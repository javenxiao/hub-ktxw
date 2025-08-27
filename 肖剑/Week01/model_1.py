# 朴素贝叶斯模型特点：
#   1、计算效率高，特别适合文本分类
#   2、基于概率理论，假设特征条件独立
#   3、自动处理多分类问题
import jieba    #中文分词库
import pandas as pd #数据处理库
from sklearn.feature_extraction.text import CountVectorizer #文本特征提取
from sklearn.naive_bayes import MultinomialNB  # 导入朴素贝叶斯

# 数据加载
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
#打印前五行数据预览
print(dataset.head(6))

# 文本预处理
input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 特征提取
vector = CountVectorizer()  # 创建词袋模型转换器 默认是使用标点符号分词
vector.fit(input_sentences.values)  #fit() 学习词汇表（所有出现的词语）
input_feature = vector.transform(input_sentences.values)    #transform() 将文本转换为词频特征矩阵（稀疏矩阵）

# 使用朴素贝叶斯模型
nb_model = MultinomialNB()  # 创建多项式朴素贝叶斯分类器
nb_model.fit(input_feature, dataset[1].values)  # 训练模型

# 预测新文本
test_query = "帮我完成这个工作任务"
test_sentence = " ".join(jieba.lcut(test_query))    #相同分词处理：分词后用空格连接
test_feature = vector.transform([test_sentence])    #转换为特征向量（使用训练时相同的词汇表）

print("待预测的文本:", test_query)
print("朴素贝叶斯预测结果:", nb_model.predict(test_feature))
print("预测概率:", nb_model.predict_proba(test_feature))