import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 数据加载与预处理
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 提取文本列并转换为列表
string_labels = dataset[1].tolist()  # 提取标签列并转换为列表

# 创建标签到索引的映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将字符串标签转换为数字标签
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符到索引的映射，初始化包含填充符
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)  # 为新字符分配索引

# 创建索引到字符的反向映射
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)  # 计算词汇表大小

max_len = 40  # 设置文本最大长度


# 2. 自定义数据集类
class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为张量
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        text = self.texts[idx]  # 获取指定索引的文本
        # 将文本转换为索引序列，未知字符使用0(填充符)
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 填充序列到最大长度
        indices += [0] * (self.max_len - len(indices))
        # 返回索引序列和对应的标签
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# 3. 定义GRU模型类
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        # 嵌入层：将字符索引转换为密集向量表示
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU层：处理序列数据，捕获时序依赖关系
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层：将GRU的最终隐藏状态映射到输出类别
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 嵌入层：将索引转换为嵌入向量
        # 输入x形状: [batch_size, seq_len]
        # 输出embedded形状: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # GRU层处理序列
        # gru_out形状: [batch_size, seq_len, hidden_dim] (所有时间步的输出)
        # hidden形状: [1, batch_size, hidden_dim] (最后一个时间步的隐藏状态)
        gru_out, hidden = self.gru(embedded)

        # 使用最后一个时间步的隐藏状态进行分类
        # 移除额外的维度并应用全连接层
        out = self.fc(hidden.squeeze(0))
        return out


# 4. 数据准备和模型初始化
# 创建数据集实例
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
# 创建数据加载器，设置批量大小和是否打乱数据
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

# 设置模型参数
embedding_dim = 64  # 嵌入向量维度
hidden_dim = 128  # GRU隐藏状态维度
output_dim = len(label_to_index)  # 输出类别数量

# 创建GRU模型实例
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# 定义损失函数（交叉熵损失）
criterion = nn.CrossEntropyLoss()
# 定义优化器（Adam优化器）
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练循环
num_epochs = 4  # 训练轮数
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 累计损失

    # 遍历数据加载器中的每个批次
    for idx, (inputs, labels) in enumerate(dataloader):
        # inputs形状: [batch_size, max_len] (字符索引序列)
        # labels形状: [batch_size] (类别标签)

        optimizer.zero_grad()  # 清零梯度，防止梯度累积

        # 前向传播
        outputs = model(inputs)  # 输出形状: [batch_size, output_dim]

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播，计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 累计损失
        running_loss += loss.item()

        # 每50个批次打印一次损失
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印每个epoch的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 6. 预测函数
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    # 将文本转换为索引序列
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充序列到最大长度
    indices += [0] * (max_len - len(indices))
    # 添加批次维度，形状从 [max_len] 变为 [1, max_len]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        output = model(input_tensor)  # 前向传播，获取预测结果

    # 获取预测类别索引
    _, predicted_index = torch.max(output, 1)  # 找到最大值的索引
    predicted_index = predicted_index.item()  # 转换为Python标量
    # 将索引映射回标签
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 7. 使用训练好的模型进行预测
# 创建索引到标签的反向映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 对新文本进行预测
new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")