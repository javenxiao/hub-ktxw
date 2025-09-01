import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 数据加载和预处理
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
labels = dataset[1].tolist()

# 创建标签映射
label_to_index = {label: i for i, label in enumerate(set(labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in labels]

# 创建字符映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40


# 数据集类
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.bow_vectors = self._create_bow_vectors(texts, char_to_index, max_len, vocab_size)

    def _create_bow_vectors(self, texts, char_to_index, max_len, vocab_size):
        bow_vectors = []
        for text in texts:
            # 创建词袋向量
            bow_vector = torch.zeros(vocab_size)
            for char in text[:max_len]:
                if char in char_to_index:
                    bow_vector[char_to_index[char]] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 不同结构的模型
def create_model(model_type, input_dim, hidden_dim, output_dim):
    if model_type == "simple":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    elif model_type == "deep":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    elif model_type == "wide":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )


# 训练函数
def train_model(model, dataloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return losses


# 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 训练不同模型
output_dim = len(label_to_index)
models_config = [
    ("simple", 64, "简单模型-隐藏层64"),
    ("simple", 128, "简单模型-隐藏层128"),
    ("deep", 128, "深层模型-隐藏层128"),
    ("wide", 128, "宽层模型-隐藏层128")
]

results = {}
for model_type, hidden_dim, name in models_config:
    print(f"\n训练 {name}")
    model = create_model(model_type, vocab_size, hidden_dim, output_dim)
    losses = train_model(model, dataloader)
    results[name] = losses

# 绘制结果
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name, marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的Loss变化')
plt.legend()
plt.grid(True)
plt.savefig('model_comparison.png')
plt.show()


# 预测函数
def predict(text, model, char_to_index, vocab_size, index_to_label):
    bow_vector = torch.zeros(vocab_size)
    for char in text:
        if char in char_to_index:
            bow_vector[char_to_index[char]] += 1

    model.eval()
    with torch.no_grad():
        output = model(bow_vector.unsqueeze(0))
        predicted = torch.argmax(output, 1).item()

    return index_to_label[predicted]


# 使用最后一个模型进行预测
test_texts = ["帮我导航到北京", "查询明天北京的天气"]
for text in test_texts:
    prediction = predict(text, model, char_to_index, vocab_size, index_to_label)
    print(f"输入: '{text}' -> 预测: '{prediction}'")