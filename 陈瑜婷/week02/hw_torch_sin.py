import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)  # 从-2π到2π的200个点
y_numpy = np.sin(X_numpy) + np.random.randn(200, 1) * 0.1  # 加上一些噪声

# 将NumPy数组转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


# 定义模型
# nn.Linear(in_features, out_features)
# 在这里，输入和输出的特征数都为1
network = nn.Sequential(
            # 第1层：从 1 到 50
            nn.Linear(1,50),
            nn.Tanh(), # 增加模型的复杂度，非线性

            # 第2层：从 50 到 50
            nn.Linear(50, 50),
            nn.Tanh(),

            # 输出层：从 50 到 1
            nn.Linear(50, 1)
        )

# 定义损失函数 (均方误差)
loss_fn = nn.MSELoss()

# 定义优化器 (随机梯度下降)
# model.parameters() 会自动找到模型中需要优化的参数（即a和b）
optimizer = torch.optim.Adam(network.parameters(), lr=0.001) # lr 是学习率

# 训练模型
num_epochs = 1000  # 训练迭代次数
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = network(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化：
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 将模型切换到评估模式
network.eval()

# 禁用梯度计算，因为我们不再训练
with torch.no_grad():
    y_predicted = network(X).numpy() # 使用训练好的模型进行预测

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Model: y', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
