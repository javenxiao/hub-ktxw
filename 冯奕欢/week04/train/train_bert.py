import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from config import MODEL_PATH, WEIGHT_PATH

# 读取数据
data = pd.read_csv('../assets/datasets/waimai_10k.csv')

# 输入 - 文字
reviews = data['review']
print(reviews.head())
# 输出 - 标签
labels = data['label']
print(labels.head())

# 数据太多 获取部分数据拼接
# reviews = list(reviews.values[:2000]) + list(reviews.values[6000:8000])
# labels = list(labels.values[:2000]) + list(labels.values[6000:8000])
reviews = list(reviews.values)
labels = list(labels.values)

# 拆分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(
    reviews,
    labels,
    test_size=0.2,
    random_state=100,
    stratify=labels
)


# 定义数据集
class BertClassificationDataset(Dataset):

    def __init__(self, x_encoding, y):
        self.x_encoding = x_encoding
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        items = {key: torch.tensor(value[idx]) for key, value in self.x_encoding.items()}
        items['labels'] = torch.tensor(self.y[idx])
        return items


# 分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# 输入编码
train_x_encoding = tokenizer(train_x, padding=True, truncation=True, max_length=64)
test_x_encoding = tokenizer(test_x, padding=True, truncation=True, max_length=64)

# 数据集
train_dataset = BertClassificationDataset(train_x_encoding, train_y)
test_dataset = BertClassificationDataset(test_x_encoding, test_y)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Bert模型
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
# Bert模型已经包括损失函数处理
# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练模型
model.train()
epochs = 1
for epoch in range(epochs):
    count = 0
    for items in train_loader:
        # 参数
        input_ids = items['input_ids']
        attention_mask = items['attention_mask']
        labels = items['labels']
        # 前向传播 计算结果
        model_result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # 损失
        loss = model_result.loss
        # 清除梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 更新参数
        optimizer.step()
        # 打印
        count += 1
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, count, len(train_loader), loss.item()))

# 保存模型
torch.save(model.state_dict(), WEIGHT_PATH)
print(f"模型已保存至 {WEIGHT_PATH}")


# 精度计算函数
def flat_accuracy(predicts, labels):
    # 获取预测结果的最高概率索引
    predicts_flat = np.argmax(predicts, axis=1).flatten()
    # 展平真实标签
    labels_flat = labels.flatten()
    print(predicts_flat, " == ", labels_flat)
    # 计算准确率
    return np.sum(predicts_flat == labels_flat) / len(labels_flat)


# 评估模型
model.eval()
with torch.no_grad():
    count = 0
    total_eval_accuracy = 0
    for items in test_loader:
        input_ids = items['input_ids']
        attention_mask = items['attention_mask']
        labels = items['labels']
        # 前向传播 计算结果
        model_result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # 损失
        loss = model_result.loss
        # 打印
        count += 1
        print('Step [{}/{}], Loss: {:.4f}'.format(count, len(test_loader), loss.item()))
        # 获取结果
        logits = model_result.logits
        predicts = logits.detach().numpy()
        labels = labels.numpy()
        predict_accuracy = flat_accuracy(predicts, labels)
        print('Predict accuracy on test set: {:.4f}'.format(predict_accuracy))
        total_eval_accuracy += predict_accuracy
    accuracy = total_eval_accuracy / count
    print('Accuracy on test set: {:.4f}'.format(accuracy))

