import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class OneHiddenSimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(OneHiddenSimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TwoHiddenSimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(TwoHiddenSimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class ThreeHiddenSimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(ThreeHiddenSimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

# hidden_dim = 1024
# output_dim = len(label_to_index)
# model = OneHiddenSimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
# criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

hidden_list = [128, 256, 512]
model_list = [
    OneHiddenSimpleClassifier,
    TwoHiddenSimpleClassifier,
    ThreeHiddenSimpleClassifier,
]

for i, modelClass in enumerate(model_list):
    for j, hidden in enumerate(hidden_list):
        output_dim = len(label_to_index)
        model = modelClass(vocab_size, hidden, output_dim) # 维度和精度有什么关系？
        criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        num_epochs = 10
        for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
            model.train()
            running_loss = 0.0
            for idx, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if epoch == 9 :
                print(f" 隐藏层个数：{i + 1} , 隐藏层大小: {hidden},损失: {running_loss / len(dataloader):.4f}")



# 隐藏层个数：1 , 隐藏层大小: 128,损失: 0.5776
# 隐藏层个数：1 , 隐藏层大小: 256,损失: 0.5849
# 隐藏层个数：1 , 隐藏层大小: 512,损失: 0.5854
# 隐藏层个数：2 , 隐藏层大小: 128,损失: 0.6256
# 隐藏层个数：2 , 隐藏层大小: 256,损失: 0.6079
# 隐藏层个数：2 , 隐藏层大小: 512,损失: 0.6054
# 隐藏层个数：3 , 隐藏层大小: 128,损失: 1.6778
# 隐藏层个数：3 , 隐藏层大小: 256,损失: 1.6106
# 隐藏层个数：3 , 隐藏层大小: 512,损失: 1.4993
# 随着隐藏层个数和隐藏层增大，损失也增大？