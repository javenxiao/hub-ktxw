#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Author: Marky
 Time: 2025/8/27 22:32 
 Description:
 1、 理解rnn、lstm、gru的计算过程（面试用途），阅读官方文档 ：https://docs.pytorch.org/docs/2.4/nn.html#recurrent-layers
 最终 使用 GRU 代替 LSTM 实现05_LSTM文本分类.py。

2、阅读项目计划书  + 初步项目代码，写清楚四个模型的优缺点，形成一个word/markdown提交。
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../../week01/dataset.csv", sep="\t", header=None)
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
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# --- NEW LSTM Model Class ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU期望输入的形状是(seq_len, batch_size, input_size)
        # 加上batch_first=True，期望输入的形状是 (batch_size, seq_len, input_size)
        self.gru = nn.GRU(embedding_dim,hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hn = self.gru(embedded)
        #output = output[:,-1,:]
        #out = self.fc(output.squeeze(0))
        out = self.fc(hn.squeeze(0)) # n.squeeze(0)
        return out

# --- Training and Prediction ---
lstm_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)
# DataLoader 会自动将多个样本（这里是32个）批量处理成一个批次。对于序列数据，默认行为是将形状为 (seq_len,) 的单个样本批量处理成形状为 (batch_size, seq_len) 的张量。
print('dataloader',dataloader)

h0 = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, h0, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # for idx, (inputs, labels) in enumerate(dataloader):
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

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    print("Output shape:", output.shape)
    print("Output:", output)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


