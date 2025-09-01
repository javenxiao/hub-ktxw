import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
作业2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
    生成 2*sin(X)+1 并添加 0.1 噪声的模拟数据，然后构建一个包含2层隐藏层的网络模型（节点数分别为64、32）
    进行训练拟合，并绘制原始数据及模型预测数据
"""


def generate_sin_data(n_samples=1000, noise=0.1):
    """
    生成sin函数数据
    :param n_samples: 生成数据数量
    :param noise: 添加噪声
    :return: X y tensor数据集
    """
    X_numpy = np.linspace(-3 * np.pi, 3 * np.pi, n_samples)
    y_numpy = 2 * np.sin(X_numpy).ravel() + 1 + np.random.normal(0, noise, n_samples)

    X = torch.from_numpy(X_numpy).float().unsqueeze(1)  # torch 中 所有的计算 通过tensor 计算
    y = torch.from_numpy(y_numpy).float().unsqueeze(1)

    return X, y


class SinModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SinModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out


def train_data(X, y, model):
    """
    使用指定模型对数据集进行训练
    :param X: 训练集特征数据
    :param y: 训练集目标标签
    :param model: 模型
    :return: None
    """
    criterion = nn.MSELoss()  # 损失函数 内部自带激活函数，softmax
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    model.train()
    num_epochs = 5000
    loss = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if epoch % 100 == 0:
            print(f"当前Epoch {epoch}, Loss: {loss.item()}")

    print(f"训练完成后 Loss: {loss.item()}")


# 可视化结果
def plot_results(X, y_true, y_pred):
    plt.scatter(X, y_true, s=5, label='True values', alpha=0.6)
    plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel('X')
    plt.ylabel('2*sin(X)+1')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # 生成训练数据集
    X, y = generate_sin_data()

    # 构建多层网络模型并进行训练
    model = SinModel(input_dim=1, hidden_dim1=64, hidden_dim2=32, output_dim=1)
    train_data(X, y, model)

    # 预测数据，拟合sin函数
    model.eval()
    with torch.no_grad():
        y_pred = model(X)

    # 绘制结果
    plot_results(X, y, y_pred)


if __name__ == "__main__":
    main()
