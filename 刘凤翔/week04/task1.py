#数据预处理：加载CSV数据，清洗文本，划分训练/验证集。
# BERT微调：使用Hugging Face的transformers库加载预训练BERT模型，并在外卖评论数据上进行微调。
# 模型保存：将训练好的模型保存到本地。
# FastAPI部署：使用FastAPI构建一个简单的HTTP服务，提供预测接口。
# 环境准备
# 安装所需库：
# pip install transformers torch pandas scikit-learn fastapi uvicorn

# 步骤1：数据预处理
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv("waimai_10k.csv")

# 简单查看
print(df.head())

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

# 步骤2：使用BERT进行微调
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# 定义自定义Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 构建Dataset
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 开始训练
trainer.train()

# 保存模型和tokenizer
model.save_pretrained('./bert_waimai')
tokenizer.save_pretrained('./bert_waimai')

# 步骤3：使用FastAPI部署服务
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载训练好的模型和tokenizer
model_path = './bert_waimai'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)


# 定义请求体
class ReviewRequest(BaseModel):
    text: str


# 初始化FastAPI
app = FastAPI()


@app.post("/predict")
def predict(request: ReviewRequest):
    # 编码输入文本
    encoding = tokenizer(
        request.text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    # 预测
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    # 返回结果
    return {"prediction": pred, "text": request.text}

# 启动服务：uvicorn app:app --reload

# 步骤4：测试API
# import requests
#
# response = requests.post(
#     "http://127.0.0.1:8000/predict",
#     json={"text": "很好吃，送餐很快！"}
# )
# print(response.json())
