import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from 作业1_model import My_model
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
data = pd.read_csv("dataset.csv",sep="\t",header=None)
dataset = data[0].tolist()
labels = data[1].tolist()
label_to_index = {label:i for i,label in enumerate(set(labels))}
index_to_label = {i:label for label,i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in labels]
char_to_index={'<pad>': 0}
for text in dataset:
    for char in text:
        if char not in char_to_index:
            char_to_index[char]=len(char_to_index)
index_to_char = {i:char for char,i in char_to_index.items()}
size = len(char_to_index)
max_len = 40
token_texts = []
for text in dataset:
    token_text = [char_to_index.get(char,0) for char in text[:max_len]]
    token_text += [0]*(max_len-len(token_text))
    token_texts.append(token_text)

class BowDataset(Dataset):
    def __init__(self,labels,token_texts,size):
        self.labels = torch.tensor(labels,dtype=torch.long)
        bow_vectors =[]
        for token_text in token_texts:
            bow_vector = torch.zeros(size)
            for index in token_text:
                if index!=0:
                    bow_vector[index]+=1
            bow_vectors.append(bow_vector)
        self.bow_vectors = torch.stack(bow_vectors)
    def __getitem__(self, idx):
        return self.bow_vectors[idx],self.labels[idx]
    def __len__(self):
        return len(self.labels)
input_dim = size
out_dim = len(label_to_index)
model_dir = {"model_1":[64],"model_2":[128,64],"model_3":[256,128,64],"model_4":[512,256,128,64]}
train_data,test_data,train_label,test_label = train_test_split(token_texts,numerical_labels,test_size=0.2,random_state=42)
train_dataset = BowDataset(train_label,train_data,size)
test_dataset =  BowDataset(test_label,test_data,size)
train_dataloader =DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)
plt.figure(figsize=(10,6))
criterion = nn.CrossEntropyLoss()
num_epochs = 30
for num,hidden_layers in enumerate(model_dir.values()):
    train_losses = []
    test_losses = []
    model = My_model(input_dim,out_dim,hidden_layers)
    optimzer = optim.Adam(model.parameters(), lr=0.001)
    model.cuda()
    for i in range(num_epochs):
        #模型训练
        model.train()
        train_loss =0
        for idx,(input,label) in enumerate(train_dataloader):
            input = input.cuda()
            optimzer.zero_grad()
            output = model(input)
            label = label.cuda()
            a_loss = criterion(output,label)
            a_loss.backward()
            optimzer.step()
            train_loss += a_loss.item()
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {a_loss.item()}")
        train_losses.append(train_loss/len(train_dataloader))
        print(f"Epoch [{i + 1}/{num_epochs}], train_Loss: {train_loss / len(train_dataloader):.4f}")
    plt.subplot(2,2,num+1)
    epochs = range(1,num_epochs+1)
    plt.plot(epochs,train_losses,label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'隐藏层层数：{len(hidden_layers)}，每层神经单元数：{hidden_layers}')
    plt.legend()
plt.tight_layout()
plt.show()
