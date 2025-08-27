import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 读取数据集
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 将字符串标签转换为数字标签，便于模型处理
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符到索引的映射，为每个字符分配唯一的索引
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的映射（反向映射）
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 自定义数据集类，设置最大序列长度为40，用于统一所有输入文本的长度
max_len = 40
class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 将字符转换为索引
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 填充到最大长度，不足的填0
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW LSTM Model Class ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)    # 嵌入层：将字符索引转换为密集向量表示
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)    # GRU层：处理序列数据，捕捉文本中的时序模式
        self.fc = nn.Linear(hidden_dim, output_dim)     # 全连接层：将LSTM的隐藏状态映射到输出类别

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)  # gru层处理序列向量
        out = self.fc(hidden_state.squeeze(0))
        return out

# --- Training and Prediction ---
# 创建数据加载器
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# 初始化模型、损失函数和优化器
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 循环训练
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 预测函数
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")