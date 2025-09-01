import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 数据读取和预处理 (保持不变) ---
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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

class CharGRUDataset(Dataset):  # 数据集类名称稍作修改以更准确，功能不变
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- 新的 GRU 模型类 ---
class GRUClassifier(nn.Module):  # 类名改为 GRUClassifier
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 将 nn.LSTM 替换为 nn.GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # GRU 只有隐藏状态，没有细胞状态
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # GRU 返回 output 和 final hidden state
        # output: (batch_size, seq_len, hidden_dim) 包含每个时间步的隐藏状态
        # hidden: (num_layers, batch_size, hidden_dim) 是序列最后一个时间步的隐藏状态
        gru_out, hidden = self.gru(embedded)
        # 取最后一层的最后一个时间步的隐藏状态进行分类
        # 如果 num_layers=1, hidden 是 (1, batch_size, hidden_dim)，用 squeeze(0) 移除第一维
        # 如果多层，则 hidden[-1] 获取最后一层的隐藏状态
        if hidden.size(0) == 1:
            out = self.fc(hidden.squeeze(0))
        else:
            out = self.fc(hidden[-1]) # 取RNN最后一层的隐藏状态
        return out

# --- 训练和预测 (结构保持不变，只需将 LSTM 替换为 GRU) ---
# 创建 Dataset 和 DataLoader
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len) # 使用GRU数据集
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

# 定义超参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_layers = 1  # 可以尝试堆叠更多层
dropout = 0.2   # 如果在多层GRU中可以使用dropout

# 实例化模型、损失函数和优化器
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{idx}/{len(dataloader)}], Current Batch Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# 预测函数 (保持不变，因为模型接口保持一致)
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0) # (1, seq_len)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 创建 index_to_label 映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 对新文本进行预测
new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label) # 调用GRU分类函数
print(f"输入 '{new_text}' 的预测类别为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label) # 调用GRU分类函数
print(f"输入 '{new_text_2}' 的预测类别为: '{predicted_class_2}'")
