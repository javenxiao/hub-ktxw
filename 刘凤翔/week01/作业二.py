import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv('/mnt/dataset.csv', header=None)

# 将数据拆分为文本和标签两部分
texts = data[0].apply(lambda x: x.split('\t')[0])
labels = data[0].apply(lambda x: x.split('\t')[1])

# 使用 TF-IDF 对文本进行向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 构建并训练朴素贝叶斯分类器
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# 在测试集上进行预测
nb_predictions = nb_classifier.predict(X_test)

# 计算朴素贝叶斯分类器的准确率
nb_accuracy = accuracy_score(y_test, nb_predictions)

# 构建并训练支持向量机分类器
svm_classifier = LinearSVC()
svm_classifier.fit(X_train, y_train)

# 在测试集上进行预测
svm_predictions = svm_classifier.predict(X_test)

# 计算支持向量机分类器的准确率
svm_accuracy = accuracy_score(y_test, svm_predictions)

print(f'朴素贝叶斯分类器的准确率: {nb_accuracy}')
print(f'支持向量机分类器的准确率: {svm_accuracy}')