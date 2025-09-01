import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

# 加载数据
data = pd.read_csv('dataset.csv', sep='\t', header=None)

# 输入文字列表
texts = data[0]
# 标签列表
labels = data[1]

# 标签数据处理
label_to_index = {char: index for index, char in enumerate(set(labels))}
print(label_to_index)
index_to_label = {index: char for char, index in label_to_index.items()}
print(index_to_label)
# out标签数字化 - 深度学习使用数值类型
label_numbers = [label_to_index[char] for char in labels]
print(label_numbers)
output_results = torch.tensor(label_numbers)
output_size = len(label_to_index)

# 输入文字处理
# 构建词典
# <PAD>主要为了占位置
char_to_index = {'<PAD>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            # 编号为字典长度 每加1个 编号加1
            char_to_index[char] = len(char_to_index)
print(char_to_index)
index_to_char = {index: char for char, index in char_to_index.items()}
print(index_to_char)
feature_size = len(char_to_index)
# 文本token化 最大取40个字
# 文字Token化
max_len = 40


def get_text_token(text):
    """
    生成输入文字 text token
    :param text: 输入文字
    :return: 输入文字 text token
    """
    text_token = [char_to_index[char] for char in text[:max_len]]
    text_token += [0] * (max_len - len(text_token))
    return text_token


text_tokens = []
for text in texts:
    text_token = get_text_token(text)
    text_tokens.append(text_token)
print(text_tokens[0])


# 输入文字矩阵特征
def get_text_feature(text_token):
    """
    生成输入文字特征
    :param text_token: 输入文字 text token
    :return: 输入文字特征
    """
    text_feature = torch.zeros(feature_size)
    for token in text_token:
        if token != 0:
            text_feature[token] += 1
    return text_feature


text_features = []
for text_token in text_tokens:
    text_feature = get_text_feature(text_token)
    text_features.append(text_feature)
print(text_features[0])
# 拼接输入特征
input_features = torch.stack(text_features)
print(input_features.shape)


# 自定义数据集
class TextDataset(Dataset):

    def __init__(self, input_features, output_results):
        self.input_features = input_features
        self.output_results = output_results

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        return self.input_features[index], self.output_results[index]


# 数据加载器
data_set = TextDataset(input_features, output_results)
data_loader = DataLoader(
    dataset=data_set,
    batch_size=1000,
    shuffle=True
)


class OneClassier(nn.Module):
    """
    1层分类模型
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OneClassier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return output


class TwoClassier(nn.Module):
    """
    2层分类模型
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(TwoClassier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


class ThreeClassier(nn.Module):
    """
    3层分类模型
    """

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(ThreeClassier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.fc4(output)
        return output


def train(classier, data_loader, loss_fun):
    """
    训练数据
    :param classier: 分类模型
    :param data_loader: 数据加载器
    :param loss_fun: 损失函数
    :return: None
    """
    print(f"开始训练模型 -> {classier}")
    # 优化器
    optimizer = optim.Adam(classier.parameters(), lr=0.01)
    # 模型设置训练模式
    classier.train()
    # 训练10次
    epoch_size = 10
    for epoch in range(epoch_size):
        # 每次训练集数据从data_loader数据加载器获取
        for index, (input_feature, output_result) in enumerate(data_loader):
            # 前向传播
            classier_result = classier(input_feature)
            # 计算损失
            loss = loss_fun(classier_result, output_result)
            # 清除梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度下降更新参数
            optimizer.step()
            # 打印
            if index == len(data_loader) - 1:
                print("{} loss --> : {:.4f}".format(epoch, loss.item()))


def predict(text, classier):
    """
    预测结果
    :param text:
    :param classier:
    :return:
    """
    text_token = get_text_token(text)
    text_feature = get_text_feature(text_token)
    input_feature = text_feature.unsqueeze(0)
    # 模型设置评估模式
    classier.eval()
    # 关闭梯度 获取结果
    with torch.no_grad():
        # 预测结果
        output_result = classier(input_feature)
    # 归一化
    output_result = torch.softmax(output_result, dim=1)
    # 提取结果 (概率, 索引)
    max_probs, max_indices = torch.max(output_result, 1)
    output_label = index_to_label[max_indices.item()]
    print(f'{text} --> {output_label}, {max_probs}')
    return output_label


def eval(classier, loss_fun, input_features, output_results):
    """
    评估模型
    :param classier: 分类模型
    :param loss_fun: 损失函数
    :param input_features: 输入特征
    :param output_results: 输入结果
    :return: 损失
    """
    print(f"开始评估模型 -> {classier}")
    # 模型设置评估模式
    classier.eval()
    # 关闭梯度 获取结果
    with torch.no_grad():
        # 预测结果
        classier_result = classier(input_features)
        # 计算损失
        loss = loss_fun(classier_result, output_results)
    print(f"模型{classier}损失 -> {loss.item()}")
    return loss.item()


def test(classier):
    """
    测试数据
    :param text: 输入文字
    :param classier: 分类模型
    :return: None
    """
    new_text = "帮我导航到北京"
    predicted_class = predict(new_text, classier)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = predict(new_text_2, classier)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 模型
one_classier_64 = OneClassier(feature_size, 64, output_size)
one_classier_128 = OneClassier(feature_size, 128, output_size)
one_classier_512 = OneClassier(feature_size, 512, output_size)
two_classier_32_64 = TwoClassier(feature_size, 32, 64, output_size)
three_classier_32_64_32 = ThreeClassier(feature_size, 32, 64, 32, output_size)
# 损失函数
loss_fun = nn.CrossEntropyLoss()

print(("=" * 50))

# 训练评估模型 one_classier_64
train(one_classier_64, data_loader, loss_fun)
eval(one_classier_64, loss_fun, input_features, output_results)
test(one_classier_64)

print(("=" * 50))

# 训练评估模型 one_classier_128
train(one_classier_128, data_loader, loss_fun)
eval(one_classier_128, loss_fun, input_features, output_results)
test(one_classier_128)

print(("=" * 50))

# 训练评估模型 one_classier_512
train(one_classier_512, data_loader, loss_fun)
eval(one_classier_512, loss_fun, input_features, output_results)
test(one_classier_512)

print(("=" * 50))

# 训练评估模型 two_classier_32_64
train(two_classier_32_64, data_loader, loss_fun)
eval(two_classier_32_64, loss_fun, input_features, output_results)
test(two_classier_32_64)

print(("=" * 50))

# 训练评估模型 three_classier_32_64_32
train(three_classier_32_64_32, data_loader, loss_fun)
eval(three_classier_32_64_32, loss_fun, input_features, output_results)
test(three_classier_32_64_32)

