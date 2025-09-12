import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import BertForSequenceClassification
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. 数据准备和预处理
def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    dataset = pd.read_csv(file_path, sep=",", header=0)
    
    # 假设数据集中有 'review' 和 'label' 列
    texts = list(dataset['review'].values)
    labels = list(dataset['label'].values)
    
    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=0.2, 
        stratify=labels,  # 按标签分层
        random_state=42
    )
    
    return x_train, x_test, y_train, y_test

# 2. 创建自定义数据集
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 3. 定义BERT分类模型
class BertClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', num_classes=2, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.linear(output)

# 4. 训练函数
def train_model(model, train_loader, val_loader, learning_rate=2e-5, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # 验证
        accuracy = evaluate_model(model, val_loader, device)
        print(f'Validation Accuracy: {accuracy:.4f}')
    
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return accuracy_score(actual_labels, predictions)

# 5. FastAPI应用 - 使用新的生命周期管理
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    global model, tokenizer
    print("Loading model...")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    model = BertClassifier()
    model.load_state_dict(torch.load('bert_classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    print("模型加载完成")
    yield
    # 关闭时清理
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    sentiment: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(review: ReviewRequest):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoding = tokenizer(
        review.text,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    
    sentiment = "正面" if prediction.item() == 1 else "负面"
    
    return PredictionResponse(
        prediction=prediction.item(),
        confidence=confidence.item(),
        sentiment=sentiment
    )

@app.get("/")
async def root():
    return {"message": "外卖评价情感分析API"}

# 6. 主函数
def main():
    # 加载数据
    x_train, x_test, y_train, y_test = load_and_preprocess_data("./dataset.csv")
    print(x_train)
    
    # # 初始化tokenizer和模型
    # tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    # model = BertClassifier(num_classes=2)  # 二分类问题，改为2个类别
    
    # # 创建数据加载器
    # train_dataset = ReviewDataset(x_train, y_train, tokenizer)
    # test_dataset = ReviewDataset(x_test, y_test, tokenizer)
    
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=16)
    
    # # 训练模型
    # print("开始训练模型...")
    # trained_model = train_model(model, train_loader, test_loader, epochs=3)
    
    # # 保存模型
    # torch.save(trained_model.state_dict(), 'bert_classifier.pth')
    # print("模型已保存")
    
    # 启动API服务
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()