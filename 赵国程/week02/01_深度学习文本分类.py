import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# 固定随机种子
torch.manual_seed(6)

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 创建标签索引 label:index
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 标签数组
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字表(正向，反向)
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}

# 字表大小，最大长度
vocab_size = len(char_to_index)
max_len = 40


class CharBowDataSet(Dataset):

    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        """
        句子向量化，向量化后仅保留字在本句中出现的次数，没有位置信息
        :return:
        """
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char) for char in text[:self.max_len]]
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

    def __getitem__(self, item):
        return self.bow_vectors[item], self.labels[item]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.inputLinear = nn.Linear(input_dim, hidden_dim)
        self.inputFunc = nn.Sigmoid()
        self.hiddenLinears = [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_size)]
        self.hiddenFunc = nn.ReLU()
        self.outputLinear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.inputLinear(x)
        out = self.inputFunc(out)
        for hiddenLinear in self.hiddenLinears:
            out = hiddenLinear(out)
            out = self.hiddenFunc(out)
        out = self.outputLinear(out)
        return out


# 划分数据集
full_dataset = CharBowDataSet(texts, numerical_labels, char_to_index, max_len, vocab_size)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

hidden_size = 1
hidden_dim = 64
output_dim = len(label_to_index)
num_epochs = 30
learning_rate = 0.01

model = SimpleClassifier(vocab_size, hidden_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # 训练
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataset:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataset:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    # 打印结果
    print(
        f"Epoch: {epoch + 1}/{num_epochs} "
        f"Training Loss: {train_loss / len(train_dataset):.3f} "
        f"Validation Loss: {val_loss / len(val_dataset):.3f} "
    )

    # hidden_size = 1, hidden_dim = 64, num_epochs = 10, learning_rate = 0.01,Validation Loss: 0.288
    # hidden_size = 1, hidden_dim = 64, num_epochs = 10, learning_rate = 0.05,Validation Loss: 0.334
    # hidden_size = 1, hidden_dim = 64, num_epochs = 20, learning_rate = 0.01,Validation Loss: 0.324
    # hidden_size = 1, hidden_dim = 128, num_epochs = 10, learning_rate = 0.01,Validation Loss: 0.324
    # hidden_size = 1, hidden_dim = 128, num_epochs = 10, learning_rate = 0.05,Validation Loss: 0.335
    # hidden_size = 1, hidden_dim = 128, num_epochs = 20, learning_rate = 0.01,Validation Loss: 0.284
    # hidden_size = 1, hidden_dim = 64, num_epochs = 30, learning_rate = 0.01,Validation Loss: 0.294


    # hidden_size = 2, hidden_dim = 64, num_epochs = 10, learning_rate = 0.01,Validation Loss: 0.483
    # hidden_size = 2, hidden_dim = 64, num_epochs = 10, learning_rate = 0.05,Validation Loss: 0.295
    # hidden_size = 2, hidden_dim = 64, num_epochs = 20, learning_rate = 0.01,Validation Loss: 0.353
    # hidden_size = 2, hidden_dim = 128, num_epochs = 10, learning_rate = 0.01,Validation Loss: 0.438
    # hidden_size = 2, hidden_dim = 128, num_epochs = 10, learning_rate = 0.05,Validation Loss: 0.284
    # hidden_size = 2, hidden_dim = 128, num_epochs = 20, learning_rate = 0.01,Validation Loss: 0.328
    # hidden_size = 2, hidden_dim = 128, num_epochs = 30, learning_rate = 0.01,Validation Loss: 0.296

    # 总结：
    # 模型越复杂，学习率越低，要达到比较好的效果所需要的训练量就越多
    # 学习率较高时，可以快速达到效果，但准确度上限低，但增加模型复杂度可以提高准确度
    # 神经元数量越多，网络层数越多，准确率上限越高，但收敛速度慢
