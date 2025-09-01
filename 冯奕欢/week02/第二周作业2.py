# 任务：构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化

import math
import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 生成大致符合sin函数随机数据
# 设置随机种子 保证每次生成都是一样的随机数
np.random.seed(100)
point_size = 2000
X = np.sort(np.random.uniform(low=0, high=2*2*np.pi, size=point_size))
y = np.sin(X) + 0.1 * np.random.randn(point_size)

# Numpy转Tensor unsqueeze可以扩展维度 从[2000]转为[1, 2000]
X = torch.from_numpy(X).float().unsqueeze(1)
y = torch.from_numpy(y).float().unsqueeze(1)
print(X.shape)
print(y.shape)


# 自定义模型
class SinModule(nn.Module):

    def __init__(self):
        super(SinModule, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 1280)
        self.fc3 = nn.Linear(1280, 1280)
        self.fc4 = nn.Linear(1280, 128)
        self.fc5 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.fc4(output)
        output = self.relu(output)
        output = self.fc5(output)
        return output


# 模型
module = SinModule()
# 损失函数
loss_fun = nn.MSELoss()
# 优化器
optimizer = torch.optim.Adam(module.parameters(), lr=0.002)

# 设置模型为训练模式
module.train()

# 开始训练
epoch_size = 1000
for epoch in range(epoch_size):
    # 前向传播
    module_result = module(X)
    # 计算损失
    loss = loss_fun(module_result, y)
    # 清除梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 打印损失
    if epoch % 100 == 0:
        print(f"{epoch} loss --> {loss.item()}")

# 预测模式
module.eval()

# 关闭梯度 预测结果
with torch.no_grad():
    predict_y = module(X)
    loss = loss_fun(predict_y, y)
    print(f"last loss ---> {loss.item()}")

# 可视化
plt.figure()
plt.scatter(X, y, alpha=0.8, color='blue', label='sin point')
plt.plot(X, predict_y, color='red', label='sin line')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
