import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class CharLSTMDataset(Dataset):
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

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super(GRUClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)

        gru_output, hidden_state = self.gru(embedded)

        if self.gru.num_layers > 1:
            last_hidden = hidden_state[-1]
        else:
            last_hidden = hidden_state.squeeze(0)

        last_hidden = self.dropout(last_hidden)

        output = self.fc(last_hidden)

        return output

gru_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
gru_dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_layers = 2
dropout = 0.2

gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加权重衰减

num_epochs = 4
for epoch in range(num_epochs):
    gru_model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(gru_dataloader):
        gru_optimizer.zero_grad()
        outputs = gru_model(inputs)
        loss = gru_criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(gru_model.parameters(), max_norm=1.0)

        gru_optimizer.step()
        running_loss += loss.item()

        if idx % 50 == 0:
            print(f"Epoch {epoch + 1}, Batch {idx}, Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(gru_dataloader):.4f}")

def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_label = index_to_label[predicted_index.item()]

    return predicted_label

new_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的歌",
    "设置明天早上8点的闹钟"
]

print("GRU模型预测结果:")
for text in new_texts:
    predicted = classify_text_gru(text, gru_model, char_to_index, max_len, index_to_label)
    print(f"输入: '{text}' -> 预测: '{predicted}'")
