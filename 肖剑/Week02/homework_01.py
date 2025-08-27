"""
1⃣️【第二周作业】
1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 标签编码，将字符串标签转换为数字索引
# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}   # 字典推导式：set() 函数将列表转换为集合，自动去除重复项, enumerate()函数遍历集合中的每个元素，同时返回索引值和元素值
numerical_labels = [label_to_index[label] for label in string_labels]       # 创建新列表的简洁语法

# 构建字符级别词汇表，NLP的重要性，文本数值化，内存效率相比one-hot编码更节省内存，可以处理任意语言的文本，为后续词袋模型或神经网络提供数值化的输入数据
char_to_index = {'<pad>': 0}    # 填充字符索引为0，<pad>是填充标记，用于使所有文本长度一致
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)    # 分配新索引， len()能巧妙的遍历赋值

# 创建反向映射，从索引到字符的映射，便于后续编码
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
# print(char_to_index)

max_len = 40

"""
词袋模型的特点
优点：
    简单高效：计算复杂度低
    固定长度：所有文本都转换为相同长度的向量
    忽略顺序：只关注字符出现次数，不关注顺序
缺点：
    丢失顺序信息："你好"和"好你"会有相同的表示
    稀疏性：对于大词汇表，向量会很稀疏
    语义信息有限：无法捕捉字符间的语义关系
"""
#数据集类定义,用于将文本数据转换为词袋向量表示
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)    # 转换为tensor的标签
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()   # 创建词袋向量

    def _create_bow_vectors(self):
        # 将文本转换为索引序列并进行填充
        tokenized_texts = []
        for text in self.texts:
            # 将文本转换为索引，最多去max_len个字符    text[:self.max_len] 截取文本，确保不超过最大长度
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 填充到max_len长度（用0填充）
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        # 创建词袋向量
        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)   # 创建全零向量
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)     # 将所有向量堆叠成张量

    # 必需的数据集方法
    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]  # 返回指定索引的数据和标签


# 模型定义 前馈神经网络(1层隐藏层， 128节点每层)
class SimpleClassifier_1_128(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier_1_128, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)     # 输入层到隐藏层
        self.relu = nn.ReLU()   # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)    # 隐藏层到输出层

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)       # 线性变换
        out = self.relu(out)    # 非线性激活
        out = self.fc2(out)     # 线性变换到输出维度
        return out


# 模型定义 前馈神经网络(1层隐藏层， 256节点每层)
class SimpleClassifier_1_256(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier_1_256, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)     # 输入层到隐藏层
        self.relu = nn.ReLU()   # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim)    # 隐藏层到输出层

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)       # 线性变换
        out = self.relu(out)    # 非线性激活
        out = self.fc2(out)     # 线性变换到输出维度
        return out


class DeepClassifier(nn.Module):    # 深的网络：2层隐藏层 128节点每层
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(DeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)     # 输入层到隐藏层
        self.relu = nn.ReLU()   # 激活函数
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 增加的隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)    # 隐藏层到输出层

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)       # 线性变换
        out = self.relu(out)    # 非线性激活
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)     # 线性变换到输出维度
        return out


class VeryDeepClassifier(nn.Module):  # 更深的网络: 3层隐藏层, 64节点每层
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VeryDeepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # 额外增加两层隐藏层网络
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


# 数据准备和模型初始化
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

output_dim = len(label_to_index)    # 类别数量
num_epochs = 20     # 训练轮数

# 定义不同模型数量
model_configs = [
    {"name": "1层128节点", "model_class": SimpleClassifier_1_128, "hidden_dim": 128},
    {"name": "1层256节点", "model_class": SimpleClassifier_1_256, "hidden_dim": 256},
    {"name": "2层128节点", "model_class": DeepClassifier, "hidden_dim": 128},
    {"name": "3层64节点", "model_class": VeryDeepClassifier, "hidden_dim": 64},
]

# 存储每个模型的loss历史
loss_history = {config["name"]: [] for config in model_configs}

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

#训练每个模型
for config in model_configs:
    print(f"\n训练模型: {config['name']}")
    model = config["model_class"](vocab_size, config["hidden_dim"], output_dim)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数 内部自带激活函数，softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

    epoch_losses = []
    for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()   # 设置模型为训练模式
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()       # 清零梯度
            outputs = model(inputs)     # 前向传播
            loss = criterion(outputs, labels)   # 计算损失
            loss.backward()     # 反向传播
            optimizer.step()    # 更新参数
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    loss_history[config["name"]] = epoch_losses

# 设置中文字体支持
try:
    # 尝试使用系统中已有的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告: 无法设置中文字体，图表中的中文可能显示为方框")

# 绘制loss曲线对比
plt.figure(figsize=(10, 6))
for model_name, losses in loss_history.items():
    plt.plot(range(1, num_epochs+1), losses, label=model_name, marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型架构的Loss变化对比')
plt.legend()
plt.grid(True)
plt.savefig('model_architecture_comparison.png')
plt.show()

# 打印最终loss比较
print("\n各模型最终Loss比较:")
for model_name, losses in loss_history.items():
    print(f"{model_name}: {losses[-1]:.4f}")