import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#用sin函数生成模拟数据
np.random.seed(42)  #保证每次生成的数据一样
X_numpy = np.random.uniform(-2*np.pi, 2*np.pi,(1000,1))
y_numpy = 3*np.sin(X_numpy)+np.random.randn(1000,1)
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()
#构造神经网络模型
class Mymodel(nn.Module):
    def __init__(self,input_dim,hiden_dim1,hiden_dim2,hidden_dim3,out_dim):
        super(Mymodel,self).__init__()
        self.fc1 = nn.Linear(input_dim,hiden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiden_dim1,hiden_dim2)
        self.fc3 = nn.Linear(hiden_dim2,hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3,out_dim)
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out
#实例化模型
model = Mymodel(1,8,16,8,1)
criterion = nn.MSELoss()
optimer =optim.Adam(model.parameters(),lr=0.001)
#模型训练
model.train()
for i in range(10000):
    optimer.zero_grad()
    y_pred = model(X)
    loss = criterion(y,y_pred)
    loss.backward()
    optimer.step()
    if (i+1)%100==0:
        print(f"当前epoch{i+1}/10000,loss:{loss.item():.4f}")
#生成预测数据
x_plot = np.linspace(-2*np.pi, 2*np.pi, 100)
x_plot_tensor = torch.from_numpy(x_plot).float().unsqueeze(1)
#模型预测
model.eval()
with torch.no_grad():
    y_predict = model(x_plot_tensor)
plt.figure(figsize=(10,6))
plt.scatter(x_plot, 3*np.sin(x_plot_tensor)+np.random.randn(100,1), alpha=0.6, label='Raw Data', s=10, color='green')
plt.plot(x_plot, y_predict.numpy(), label='Model Prediction', color='orange', linewidth=2)
plt.plot(x_plot, 3*np.sin(x_plot), label='True sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Sin Function Fitting')
plt.legend()
plt.show()
