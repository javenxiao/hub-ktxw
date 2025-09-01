import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# 数据集创建
class SinDataset(Dataset):
    def __init__(self, num_samples=5000, sample_range=2 * np.pi):
        self.num_samples = num_samples
        self.sample_range = sample_range
        self.items = self._generate_dataset()

    def _generate_dataset(self):
        """
        平均生成0到sample_range中的num_samples个x,sin(x)数据
        :param num_samples:
        :param sample_range:
        :return:
        """
        x = np.linspace(0, self.sample_range, self.num_samples)
        # 生成sin(x)，+1避免负数
        y = np.sin(x) + 1
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def __len__(self):
        return len(self.items[0])

    def __getitem__(self, index):
        return self.items[0][index], self.items[1][index]


# 模型定义
class SinModel(nn.Module):
    def __init__(self, input_dim, hidden_size, hidden_dim, output_dim):
        super(SinModel, self).__init__()
        self.inputLinear = nn.Linear(input_dim, hidden_dim)
        self.inputFunc = nn.Sigmoid()
        self.hiddenLinears = [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_size)]
        self.hiddenFunc = nn.ReLU()
        # 输出层这里不使用激活函数，因为输出是连续的
        self.outputLinear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.inputLinear(x)
        out = self.inputFunc(out)
        for hiddenLinear in self.hiddenLinears:
            out = hiddenLinear(out)
            out = self.hiddenFunc(out)
        out = self.outputLinear(out)
        return out


# 数据集拆分，80%训练，20%验证
full_dataset = SinDataset()
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(full_dataset, batch_size=10)
val_dataloader = DataLoader(val_dataset, batch_size=10)

# 模型参数
input_dim = 1
hidden_size = 2
hidden_dim = 8
output_dim = 1
num_epochs = 500
learning_rate = 0.05

model = SinModel(input_dim, hidden_size, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    # 训练
    for inputs, target in train_dataloader:
        # 维度转换
        inputs = inputs.view(-1, 1)
        target = target.view(-1, 1)
        optimizer.zero_grad()
        target_pred = model(inputs)
        loss = criterion(target_pred, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.view(-1, 1)
            target = target.view(-1, 1)
            target_pred = model(inputs)
            loss = criterion(target_pred, target)
            val_loss += loss.item()
    print(
        f"Epoch: {epoch + 1:2d}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}")

# 绘图
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) + 1

model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(x).view(-1, 1))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='sin(x)', color='blue', alpha=0.6)
plt.scatter(x, y_pred.numpy(), label='Model prediction', color='red', alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
