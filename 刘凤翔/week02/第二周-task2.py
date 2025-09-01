import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 生成sin函数数据
def generate_sin_data(num_samples=1000, x_range=(-3, 3)):
    x = torch.linspace(x_range[0], x_range[1], num_samples).unsqueeze(1)
    y = torch.sin(x)
    return x, y


# 创建数据集和数据加载器
x_data, y_data = generate_sin_data()
dataset = torch.utils.data.TensorDataset(x_data, y_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# 定义不同结构的模型
def create_model(model_type, input_dim=1, hidden_dim=64, output_dim=1):
    if model_type == "simple":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    elif model_type == "deep":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    elif model_type == "wide":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim)
        )
    elif model_type == "deep_wide":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


# 训练函数
def train_model(model, dataloader, num_epochs=200):
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器
    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return losses


# 训练不同模型
models_config = [
    ("simple", 32, "简单模型-隐藏层32"),
    ("simple", 64, "简单模型-隐藏层64"),
    ("deep", 32, "深层模型-隐藏层32"),
    ("wide", 32, "宽层模型-隐藏层32"),
    ("deep_wide", 32, "深度宽度模型-隐藏层32")
]

results = {}
trained_models = {}

for model_type, hidden_dim, name in models_config:
    print(f"\n训练 {name}")
    model = create_model(model_type, 1, hidden_dim, 1)
    losses = train_model(model, dataloader)
    results[name] = losses
    trained_models[name] = model

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('不同模型结构的Loss变化')
plt.legend()
plt.grid(True)
plt.savefig('sin_fitting_loss.png')
plt.show()

# 可视化拟合效果
plt.figure(figsize=(15, 10))

# 绘制真实sin曲线
x_test = torch.linspace(-3, 3, 1000).unsqueeze(1)
y_true = torch.sin(x_test)
plt.plot(x_test.numpy(), y_true.numpy(), 'k-', linewidth=2, label='True sin(x)')

# 绘制各模型的预测曲线
colors = ['r', 'g', 'b', 'c', 'm']
for i, (name, model) in enumerate(trained_models.items()):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
    plt.plot(x_test.numpy(), y_pred.numpy(),
             linestyle='--',
             color=colors[i % len(colors)],
             linewidth=1.5,
             label=f'{name}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Sin函数拟合效果比较')
plt.legend()
plt.grid(True)
plt.savefig('sin_fitting_comparison.png')
plt.show()

# 评估模型在测试集上的性能
test_x = torch.linspace(-3.5, 3.5, 1000).unsqueeze(1)
test_y = torch.sin(test_x)

print("\n模型在扩展测试集上的MSE损失:")
for name, model in trained_models.items():
    model.eval()
    with torch.no_grad():
        predictions = model(test_x)
        mse = nn.MSELoss()(predictions, test_y).item()
    print(f"{name}: {mse:.6f}")