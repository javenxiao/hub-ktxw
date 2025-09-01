import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 加载数据
data = pd.read_csv('dataset.csv', sep='\t', header=None)
texts = data[0]
labels = data[1]
print(texts.head())
print(labels.head())

# 构造词典
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
index_to_char = {index: char for char, index in char_to_index.items()}
print(char_to_index)
print(index_to_char)
vocab_size = len(char_to_index)

# 输入
max_len = 40
input_features = []
for text in texts:
    input_feature = [char_to_index[char] for char in text[:max_len]]
    input_feature += [0] * (max_len - len(input_feature))
    input_features.append(input_feature)
input_features = torch.tensor(input_features)
print(input_features.shape)

# 输出
label_to_index = {label: index for index, label in enumerate(set(labels))}
index_to_label = {index: label for index, label in enumerate(set(labels))}
print(label_to_index)
output_features = [label_to_index[label] for label in labels]
output_features = torch.tensor(output_features)
print(output_features.shape)


# 自定义数据集
class FeatureDataset(Dataset):

    def __init__(self, inputs, outputs):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# LSTM分类模型
class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # lstm层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入特征转为词向量
        x = self.embedding(x)
        # 输出隐藏状态和细胞状态
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        # 隐藏层维度降到分类结果维度
        out = self.fc(hidden_state.squeeze(0))
        return out


# GRU分类器
class GRUClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # gru层
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 输入特征转为词向量
        x = self.embedding(x)
        # 输出隐藏状态 没有细胞状态
        gru_out, hidden_state = self.gru(x)
        # 隐藏层维度降到分类结果维度
        out = self.fc(hidden_state.squeeze(0))
        return out


# 模型
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
# classifier = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
classifier = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# 损失函数
loss_fun = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.002)
# 数据加载器
data_set = FeatureDataset(input_features, output_features)
data_loader = DataLoader(
    dataset=data_set,
    batch_size=100,
    shuffle=True
)

# 训练模式
classifier.train()
# 开始训练
epoch_size = 10
for epoch in range(epoch_size):
    for index, (inputs, outputs) in enumerate(data_loader):
        # 前向传播
        classifier_result = classifier(inputs)
        # 计算损失
        loss = loss_fun(classifier_result, outputs)
        # 清除梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印损失
        if index == len(data_loader) - 1:
            print(f"{epoch} loss: {loss.item():.4f}")


def predict_text(text):
    """
    预测
    :param text: 输入内容
    :return: 分类
    """
    # 输入
    input_feature = [char_to_index[char] for char in text[:max_len]]
    input_feature += [0] * (max_len - len(input_feature))
    input_feature = torch.tensor(input_feature).unsqueeze(0)
    # 评估模式
    classifier.eval()
    # 开始评估
    with torch.no_grad():
        output_result = classifier(input_feature)
        # 归一化
        output_result = torch.softmax(output_result, dim=1)
        print(output_result)
        # 获取概率最大结果
        value, index = torch.max(output_result, dim=1)
        print(value, index)
        # 预测标签
        predict_label = index_to_label[index.item()]
        return predict_label


new_text = "帮我导航到北京"
predicted_class = predict_text(new_text)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = predict_text(new_text_2)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
