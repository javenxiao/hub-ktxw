import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# 1.读取数据
dataset = pd.read_csv("../dataset/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()
# 2.标签处理
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
# 3.创建字表
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)


# 4.创建数据集
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
        text_indices = [self.char_to_index[char] for char in text[:self.max_len]]
        text_indices += [0] * (self.max_len - len(text_indices))
        return torch.tensor(text_indices, dtype=torch.long), self.labels[idx]


# 5.模型定义
class GRUClassifier(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, rnn_layers, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.embedding(x)
        out, hn = self.gru(out)
        # 取最后一层rnn的状态
        out = self.fc(hn[-1, :, :])
        return out


# 6.数据集构建
max_len = 40
full_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
# 7.超参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(numerical_labels)
rnn_layers = 2

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, rnn_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
# 开始训练
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(
        f"Epoch: {epoch + 1}/{num_epochs}, Train Loss:{train_loss / len(train_dataset):.4f}, Val Loss:{val_loss / len(val_dataset):.4f}")

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
test_text = "放首歌"
predicted_label = classify_text_gru(test_text, model, char_to_index, max_len, index_to_label)
print(f"预测的标签为: {predicted_label}")
