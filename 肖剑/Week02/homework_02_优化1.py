import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成模拟数据 - sin函数
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)   # 创建了从-2π到2π的1000个等间距点
y_numpy = np.sin(X_numpy)

X_mean, X_std = X_numpy.mean(), X_numpy.std()   # 计算输入数据的均值和标准差，用于标准化处理
X_numpy_normalized = (X_numpy - X_mean) / X_std  # 标准化数据

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_numpy_normalized, y_numpy, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float()

print("Sin函数数据生成完成。")
print("---" * 10)


# 2. 定义神经网络模型
class SinNet(nn.Module):
    def __init__(self, hidden_size=64):
        super(SinNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)


# 创建模型实例
model = SinNet(hidden_size=64)
print("神经网络模型结构:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()  # 使用 MSE均方误差 能更好的惩罚大误差，保持可导性
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)      # Adam 能自适应学习率，快速收敛，鲁棒性

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=100
)

# 记录初始学习率
current_lr = optimizer.param_groups[0]['lr']

# 4. 训练模型
num_epochs = 5000
train_losses = []  # 记录训练损失值
val_losses = []    # 纪录验证损失值
best_loss = float('inf')
best_model_weights = None

for epoch in range(num_epochs):
    # 训练模式
    model.train()

    # 前向传播
    y_pred = model(X_train_tensor)

    # 计算训练损失
    train_loss = loss_fn(y_pred, y_train_tensor)
    train_losses.append(train_loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 评估模式
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = loss_fn(y_val_pred, y_val_tensor)
        val_losses.append(val_loss.item())

    # 学习率调度
    scheduler.step(val_loss)
    # 检查学习率是否变化
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != current_lr:
        print(f"学习率从 {current_lr:.6f} 降低到 {new_lr:.6f}")
        current_lr = new_lr

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_weights = model.state_dict().copy()

    # 每500个epoch打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 加载最佳模型权重
model.load_state_dict(best_model_weights)

# 5. 在整个数据集上评估模型
X_full = torch.from_numpy(X_numpy_normalized).float()
model.eval()
with torch.no_grad():
    y_predicted = model(X_full).numpy()

# 设置 matplotlib 使用支持 Unicode 的字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 减号而不是 Unicode 减号

# 6. 绘制结果
plt.figure(figsize=(12, 10))

# 绘制原始数据和拟合结果
plt.subplot(3, 1, 1)
plt.scatter(X_numpy, y_numpy, label='True sin(x)', color='blue', alpha=0.6, s=10)
plt.plot(X_numpy, y_predicted, label='Neural Network Approximation', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sin Function Approximation with Neural Network')
plt.legend()
plt.grid(True)

# 绘制损失曲线
plt.subplot(3, 1, 2)
plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Val Loss', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.yscale('log')  # 使用对数刻度更好地显示损失下降
plt.legend()
plt.grid(True)

# 绘制误差分布
plt.subplot(3, 1, 3)
errors = np.abs(y_numpy.flatten() - y_predicted.flatten())
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"最终训练损失: {train_losses[-1]:.6f}")
print(f"最终验证损失: {val_losses[-1]:.6f}")
print(f"最大绝对误差: {np.max(errors):.6f}")
print(f"平均绝对误差: {np.mean(errors):.6f}")