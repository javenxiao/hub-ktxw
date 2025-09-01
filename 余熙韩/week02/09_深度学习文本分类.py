import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

## 标签转索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符对应数字字典
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40



# 字符BOW数据集
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


class SimpleClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5) # 添加Dropout层，丢弃概率为0.5  会增加loss 值
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 增加一层后loss值不稳定，过拟合， 甚至会梯度爆炸
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu2(out)
        out = self.fc2(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 将处理好的字符BOW(Bag of Words)数据集按照指定的批次大小加载到模型中进行训练。

hidden_dim = 128   # 节点数增加，loss 值下降， 过拟合
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
# 定义交叉熵损失函数，适用于多分类问题
# CrossEntropyLoss会自动将模型输出的原始分数(logits)通过softmax转换为概率分布后计算损失
criterion = nn.CrossEntropyLoss()

# 定义Adam优化器，用于更新模型参数
# 参数包括：模型参数(model.parameters())，学习率(lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义训练轮数
num_epochs = 10

# 开始训练循环，遍历每一个epoch
for epoch in range(num_epochs):
    # 设置模型为训练模式，启用dropout和batch normalization等训练特有操作
    model.train()
    # 初始化运行损失为0，用于累积每轮的总损失
    running_loss = 0.0

    # 遍历数据加载器中的每一个批次数据
    # idx:批次索引, (inputs, labels):输入特征和对应的标签
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清零优化器中的梯度，防止梯度累积
        optimizer.zero_grad()
        # 将输入数据传入模型，获取输出预测结果
        outputs = model(inputs)
        # 计算当前批次的损失值
        loss = criterion(outputs, labels)
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 累积当前批次的损失值(注意使用.item()将张量转换为Python数值)
        running_loss += loss.item()
        # 每50个批次打印一次当前损失值，用于监控训练进度
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印当前epoch的平均损失值
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 定义文本分类函数，用于对新文本进行分类预测
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将输入文本转换为字符索引序列，并截断或填充至最大长度
    # 对于不在字符字典中的字符，使用索引0('<pad>')进行替换
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    # 创建字符级词袋模型(BOW)向量，初始化为全零
    bow_vector = torch.zeros(vocab_size)
    # 统计每个字符在文本中出现的次数
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 在向量的第0维添加一个批次维度，使其符合模型输入要求
    bow_vector = bow_vector.unsqueeze(0)

    # 设置模型为评估模式，禁用dropout等训练特有操作
    model.eval()
    # 禁用梯度计算，提高推理效率
    with torch.no_grad():
        # 将处理好的特征向量传入模型进行预测
        output = model(bow_vector)

    # 获取预测结果中概率最大的类别索引
    # torch.max返回最大值和对应的索引，这里我们只关心索引
    _, predicted_index = torch.max(output, 1)
    # 将张量转换为Python数值
    predicted_index = predicted_index.item()
    # 根据索引获取对应的标签名称
    predicted_label = index_to_label[predicted_index]

    # 返回预测的类别标签
    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
