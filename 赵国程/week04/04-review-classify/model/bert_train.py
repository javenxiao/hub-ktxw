import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 第一行为header
dataset_df = pd.read_csv("../dataset/waimai_10k.csv", sep=",", header=0)

labels = dataset_df['label'].values.tolist()
texts = dataset_df['review'].values.tolist()

x_train, x_test, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels
)

tokenizer = BertTokenizer.from_pretrained("../model/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("../model/bert-base-chinese", num_labels=2)

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=32, return_tensors="pt")
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=32, return_tensors="pt")

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": test_labels
})

train_args = TrainingArguments(
    output_dir="./results",                     # 模型训练结果保存目录
    num_train_epochs=3,                         # 训练总批次
    per_device_train_batch_size=128,            # 训练时每批次大小
    per_device_eval_batch_size=128,             # 评估时每批次大小
    warmup_steps=500,                           # 学习率预热步数
    weight_decay=0.01,                          # 权重衰减
    logging_dir="./logs",                       # 日志存储目录
    logging_steps=10,                           # 每10步输出一次日志, 总步数 = 总数据集大小 / 批次大小 * 批次数
    eval_strategy="epoch",                      # 评估策略, 每批次评估一次模型
    save_strategy="epoch",                      # 保存策略, 每批次保存一次模型
    load_best_model_at_end=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": (predictions == labels).mean()}


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
