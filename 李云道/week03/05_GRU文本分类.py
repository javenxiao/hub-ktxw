import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class DataGenerator(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        indices = [self.vocab.get(char, 0) for char in text[:self.max_length]]
        indices += [0] * (self.max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[index]


class TorchClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_dim):
        super(TorchClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)
        # [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_size]
        x, _ = self.gru(x)
        # 取最后一个时间步的输出进行分类
        x = x[:, -1, :].squeeze()
        predict = self.fc(x)
        return predict


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_vocab(texts):
    vocab = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    reverse_vocab = {i: char for char, i in vocab.items()}
    return vocab, reverse_vocab


def load_data(data_path):
    dataset = pd.read_csv(data_path, sep="\t", header=None)
    texts = dataset[0].tolist()
    labels = dataset[1].tolist()

    label_to_index = {label: i for i, label in enumerate(set(labels))}
    labels = [label_to_index[label] for label in labels]
    index_to_label = {i: label for label, i in label_to_index.items()}
    return texts, labels, label_to_index, index_to_label


def evaluate(model, max_length, vocab, reverse_vocab):
    def predict_class(text):
        indices = [vocab.get(char, 0) for char in text[:max_length]]
        indices += [0] * (max_length - len(indices))
        input_ids = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            predict = model(input_ids)
        # import pdb; pdb.set_trace()
        predicted_index = torch.argmax(predict, dim=-1).item()
        predicted_label = reverse_vocab[predicted_index]

        return predicted_label

    new_text = "帮我导航到北京"
    predicted_class = predict_class(new_text)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = predict_class(new_text_2)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")


def main():
    set_seed(42)
    max_length = 40
    batch_size = 32
    embedding_dim = 64
    hidden_size = 128
    data_path = "./data/dataset.csv"
    texts, labels, label_to_index, index_to_label = load_data(data_path)
    vocab, reverse_vocab = get_vocab(texts)
    output_dim = len(label_to_index)
    learning_rate = 0.001
    epochs = 5

    dg = DataGenerator(texts, labels, vocab, max_length)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=True)
    model = TorchClassifier(len(vocab), embedding_dim, hidden_size, output_dim)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for index, batch_data in enumerate(dl):
            optim.zero_grad()
            inputs, labels = batch_data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            # import pdb; pdb.set_trace()
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            # if index % 50 == 0:
            #     print(f"Batch 个数 {index}, 当前Batch Loss: {loss.item()}")
            watch_loss.append(loss.item())
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {np.mean(watch_loss):.4f}")

    evaluate(model, max_length, vocab, index_to_label)


if __name__ == '__main__':
    main()
