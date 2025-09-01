import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)


# 1. 生成数据
def generate_data(n_samples=1000):
    # 在 [0, 4π] 范围内生成数据
    x = np.linspace(0, 4 * np.pi, n_samples)
    y = np.sin(x)
    return x, y


# 2. 创建神经网络模型
class SinModel(nn.Module):
    def __init__(self, hidden_layers=None):
        super(SinModel, self).__init__()

        # 创建层列表
        if hidden_layers is None:
            hidden_layers = [64, 64, 32]
        layers = []
        input_size = 1  # 输入特征维度

        # 添加隐藏层
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())

            input_size = hidden_size

        # 添加输出层
        layers.append(nn.Linear(input_size, 1))

        # 组合所有层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 3. 训练函数
def train_model(model, x_train, y_train, x_val, y_val, epochs=1000, lr=0.01):
    # 转换为PyTorch张量
    x_train = torch.FloatTensor(x_train).view(-1, 1)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    x_valid = torch.FloatTensor(x_val).view(-1, 1)
    y_valid = torch.FloatTensor(y_val).view(-1, 1)

    # 定义损失函数和优化器
    mseloss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练历史记录
    train_losses = []
    val_losses = []

    # 训练循环
    for epoch in range(epochs):
        # 训练模式
        model.train()

        # 前向传播
        outputs = model(x_train)
        loss = mseloss(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证模式
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_valid)
            val_loss = mseloss(val_outputs, y_valid)

        # 记录损失
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    return train_losses, val_losses


# 4. 可视化
def plot_sin(model, x, y, x_test=None):
    # 转换为PyTorch张量
    x = torch.FloatTensor(x).view(-1, 1)

    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(x).numpy().flatten()

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制原始Sin函数
    plt.plot(x, y, 'b-', linewidth=2, label='Sin(x)')

    # 绘制预测结果
    plt.plot(x, y_pred, 'r-', linewidth=2, label='Pred Sin(x)')

    # 如果有测试点，绘制测试点
    if x_test is not None:
        plt.scatter(x_test, np.sin(x_test), c='g', s=50, label='Test Points')

    plt.title('Sin Function Approximation with Neural Network', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('sin(x)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.yscale('log')  # 对数尺度更好显示
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


# 主函数
def main():
    # 生成数据
    x, y = generate_data()

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42
    )

    # 6：2：2
    print(f"训练集大小: {len(x_train)}")
    print(f"验证集大小: {len(x_val)}")
    print(f"测试集大小: {len(x_test)}")

    # 创建模型
    model = SinModel(hidden_layers=[64, 128, 64, 32])

    # 训练模型
    train_losses, val_losses = train_model(
        model, x_train, y_train, x_val, y_val,
        epochs=60, lr=0.001
    )

    # 可视化训练过程
    plot_loss(train_losses, val_losses)
    plot_sin(model, x, y, x_test)

    # 在测试集上评估
    x_test = torch.FloatTensor(x_test).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        test_loss = nn.MSELoss()(y_pred, y_test)

    print(f"\n测试集MSE损失: {test_loss.item():.6f}")


if __name__ == "__main__":
    main()
