import torch
import torch.nn as nn
class My_model(nn.Module):
    def __init__(self,input_dim,out_dim,hidden_layers):
        super(My_model,self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim,hidden_layers[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(hidden_layers[-1],out_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        # 手动实现每层的计算
        out = self.network(x)
        return out
