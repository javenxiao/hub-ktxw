import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ================= 数据准备 =================
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


# ================= 模型定义 =================
class SimpleClassifier(nn.Module):
    """单隐藏层"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class DeepClassifier(nn.Module):
    """双隐藏层"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# ================= 训练函数 =================
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    return loss_history


# ================= 实验配置 =================
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)
num_epochs = 15

experiments = {
    "1层-64节点": SimpleClassifier(vocab_size, 64, output_dim),
    "1层-128节点": SimpleClassifier(vocab_size, 128, output_dim),
    "1层-256节点": SimpleClassifier(vocab_size, 256, output_dim),
    "2层-128->64节点": DeepClassifier(vocab_size, 128, output_dim),
}

results = {}

# ================= 实验 =================
for name, model in experiments.items():
    print(f"\n===== 开始训练 {name} =====")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_history = train_model(model, dataloader, criterion, optimizer, num_epochs)
    results[name] = {
        "model": model,        # 保存模型
        "loss": loss_history   # 保存loss
    }

# ================= 绘图 =================
plt.rcParams['font.sans-serif'] = ['SimHei']   # 支持中文
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.plot(result["loss"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("不同网络结构的Loss对比")
plt.legend()
plt.grid(True)
plt.show()

# ================= 分类函数 =================
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
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


# 使用最后一个模型测试分类
index_to_label = {i: label for label, i in label_to_index.items()}
test_model = list(results.values())[-1]["model"]

new_text = "帮我导航到北京"
print(f"输入 '{new_text}' 预测为: '{classify_text(new_text, test_model, char_to_index, vocab_size, max_len, index_to_label)}'")

new_text_2 = "查询明天北京的天气"
print(f"输入 '{new_text_2}' 预测为: '{classify_text(new_text_2, test_model, char_to_index, vocab_size, max_len, index_to_label)}'")
