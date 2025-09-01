import torch
import numpy as np
import matplotlib.pyplot as plt


# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(500, 1) * 10
# y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
y_numpy = 2 * np.sin(X_numpy) + 0.2 * np.random.randn(500, 1)  # sin(x) 取值在0-1 之间，噪取值应该到合适的值，避免干扰

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b
# 这里是主要修改的部分，我们不再使用 nn.Linear。
# torch.randn() 生成随机值作为初始值。
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss()

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.SGD([a, b], lr=0.01)

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = a * torch.sin(X) + b

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"拟合的斜率 a: {a_learned:.4f}")
print(f"拟合的截距 b: {b_learned:.4f}")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    # 解决画图时X值无序的问题
    sorted_indices = torch.argsort(X, dim=0)  # 获取排序后的元素索引，生序排序
    X_sorted = X[sorted_indices].squeeze()  # 排序后的X值，squeeze 删除张量中维度大小为1的维度
    y_predicted = a_learned * torch.sin(X_sorted) + b_learned

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_sorted.numpy(), y_predicted, label=f'Model: y = {a_learned:.2f}sin(x) + {b_learned:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
