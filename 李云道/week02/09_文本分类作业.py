import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class CharBoWDataset(Dataset):
    def __init__(self, texts, label_vectors, vocab, max_length):
        self.texts = texts
        self.labels = torch.LongTensor(label_vectors)
        self.vocab = vocab
        self.max_length = max_length
        self.vocab_size = len(vocab)
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.vocab.get(char, 0) for char in text[:self.max_length]]
            # padding
            tokenized += [0] * (self.max_length - len(tokenized))
            tokenized_texts.append(tokenized)

        # 构建的词袋向量是每个文本的词频向量
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


class HiddenLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HiddenLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer(x)
        out = self.relu(out)
        return out


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layer):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # num_hidden_layer = max(1, num_hidden_layer)  # 至少一层隐藏层
        self.hidden_layer = nn.ModuleList([HiddenLayer(hidden_dim, hidden_dim) for _ in range(num_hidden_layer)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        for layer in self.hidden_layer:
            out = layer(out)
        out = self.fc2(out)
        return out


def load_data(dataset_path):
    dataset = pd.read_csv(dataset_path, sep="\t", header=None)
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()
    label_to_index = {label: i for i, label in enumerate(set(string_labels))}
    index_to_label = {v: k for k, v in label_to_index.items()}
    label_vectors = [label_to_index[label] for label in string_labels]

    return texts, label_to_index, index_to_label, label_vectors


def generate_vocabulary(texts):
    vocab = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def evaluate_classify(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


def main(num_hidden_layer):
    # 数据准备
    dataset_path = "./data/dataset.csv"
    # 文本数据，{标签: 索引}，{索引: 标签}，标签向量
    texts, label_to_index, index_to_label, label_vectors = load_data(dataset_path)
    # print(label_to_index)
    # print(index_to_label)
    vocab = generate_vocabulary(texts)

    # 超参数设置
    max_length = 40
    hidden_dim = 128
    output_dim = len(label_to_index)  # 输出维度等于标签的数量
    vocab_size = len(vocab)
    num_epochs = 10  # epoch： 将数据集整体迭代训练一次
    batch_size = 32  # batch： 数据集汇总为一批训练一次
    learning_rate = 0.01

    # 建立模型，损失函数，优化器
    train_dataset = CharBoWDataset(texts, label_vectors, vocab, max_length)  # 读取单个样本
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 读取批量数据集 -》 batch数据
    model = SimpleClassifier(vocab_size, hidden_dim, output_dim, num_hidden_layer)  # 维度和精度有什么关系
    criterion = nn.CrossEntropyLoss()  # 损失函数 内部自带激活函数，softmax
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            running_loss += loss.item()
            # if idx % 50 == 0:
            #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

        # 模型评估
        new_text = "帮我导航到北京"
        predicted_class = evaluate_classify(new_text, model, vocab, vocab_size, max_length, index_to_label)
        print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
        new_text_2 = "查询明天北京的天气"
        predicted_class_2 = evaluate_classify(new_text_2, model, vocab, vocab_size, max_length, index_to_label)
        print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


if __name__ == '__main__':
    for num_hidden_layer in [0, 1, 2, 3, 4, 5]:
        print(f"隐藏层数: {num_hidden_layer}")
        main(num_hidden_layer)
        print()
        # 隐藏层数: 1 效果最好
