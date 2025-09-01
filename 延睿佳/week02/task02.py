import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成 sin 函数数据
X_numpy = np.linspace(0, 10, 200).reshape(-1, 1)  # 自变量
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(200, 1)  # 加噪声的 sin 曲线

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层网络模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),   # 输入层 -> 隐藏层1
            nn.Tanh(),                               # 激活函数
            nn.Linear(32, 32),  # 隐藏层2
            nn.Tanh(),
            nn.Linear(32, 1)    # 输出层
        )
    def forward(self, x):
        return self.net(x)

model = MLP()

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 绘制结果
model.eval()
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data (sin + noise)', color='blue', alpha=0.5)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linestyle='--')
plt.plot(X_numpy, y_predicted, label='MLP Prediction', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
