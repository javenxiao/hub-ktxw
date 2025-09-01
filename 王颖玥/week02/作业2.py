import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
用2层，隐藏层64个节点和3层，隐藏层每层64节点两种模型拟合了
y = sinx, y = 2sin(3 * x + 1)两个正弦函数
具体结果见「说明文档」
"""

# 生成模拟数据
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 500)  # 生成有序数据
y_numpy = 2 * np.sin(3 * X_numpy + 1) + 0.1 * np.random.randn(500)
# X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 500)  # 生成有序数据
# y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(500)
X = torch.from_numpy(X_numpy).float().unsqueeze(1)
y = torch.from_numpy(y_numpy).float().unsqueeze(1)

print("数据生成完成。")
print("---" * 10)

# 2层 每层64个节点
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.tanh(self.fc1(x))
        out = self.fc2(out)
        return out

input_size = 1
hidden_dim = 64
output_dim = 1
model = SimpleClassifier(input_size, hidden_dim, output_dim)

# # 3层 每层64个节点
# class ComplexClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
#         super(ComplexClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = nn.Linear(hidden_dim2, output_dim)
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         out = self.tanh(self.fc1(x))
#         out = self.tanh(self.fc2(out))
#         out = self.fc3(out)
#         return out
#
# input_size = 1
# hidden_dim1 = 64
# hidden_dim2 = 64
# output_dim = 1
# model = ComplexClassifier(input_size, hidden_dim1, hidden_dim2, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器
# print(optimizer)

# 训练模型
num_epochs = 8000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度，不然梯度会累加
    loss.backward()        # 计算梯度
    optimizer.step()

    if (epoch + 1) % 800 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 画图
with torch.no_grad():
    y_predicted = model(X).numpy()

network_info = "2 Level, 64 Neurons"
# network_info = "3 Level, 64, 64 Neurons"
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model: y = 2 * sin(3x + 1)', color='red', linewidth=2)
plt.plot(X_numpy, 2 * np.sin(3 * X_numpy + 1), label=f'Real: y = 2 * sin(3x + 1)', color='green', linewidth=2, alpha=0.8, linestyle='--')
# plt.plot(X_numpy, y_predicted, label=f'Model: y = sinx', color='red', linewidth=2)
# plt.plot(X_numpy, np.sin(X_numpy), label=f'Real: y = sinx', color='green', linewidth=2, linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Loss Variation({network_info})')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
