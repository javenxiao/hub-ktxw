import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification,BertTokenizer,Trainer,TrainingArguments
tokenizer = AutoTokenizer.from_pretrained("../models/google-bert/bert-base-chinese")
data = pd.read_csv("./dataset/waimai_10k.csv")
dataset = data['review'].values
labels = data['label'].values
train_data,test_data,train_labels,test_labels = train_test_split(dataset,labels,test_size=0.2,random_state=42,stratify=labels)
tokenizer = BertTokenizer.from_pretrained("../models/google-bert/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("../models/google-bert/bert-base-chinese",num_labels=2)
train_encodings = tokenizer(list(train_data),truncation=True,padding=True,max_length=64)
test_encodings = tokenizer(list(test_data),truncation=True,padding=True,max_length=64)
train_dataset = Dataset.from_dict({"input_ids":train_encodings['input_ids'],
                                   "attention_mask":train_encodings['attention_mask'],
                                   "labels":train_labels})
test_dataset = Dataset.from_dict({"input_ids":test_encodings['input_ids'],
                                   "attention_mask":test_encodings['attention_mask'],
                                   "labels":test_labels})
def computer_metric(pred):
    logits,label = pred
    predict = np.argmax(logits,axis=-1)
    return {"accuracy":(predict==label).mean()}
train_args = TrainingArguments(output_dir="./result/",
                               num_train_epochs=10,
                               per_device_train_batch_size=64,
                               per_device_eval_batch_size=64,
                               warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练
                               weight_decay=0.01,                   # 权重衰减，用于防止过拟合
                               logging_dir='./logs/',                # 日志存储目录
                               logging_steps=100,                   # 每隔100步记录一次日志
                               eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
                               save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
                               load_best_model_at_end=True)
trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  compute_metrics=computer_metric)
trainer.train()
trainer.evaluate()
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    print(f"The best model is located at: {best_model_path}")
    torch.save(best_model.state_dict(), './models/weights/bert.pt')
    print("Best model saved to models/weights/bert.pt")
else:
    print("Could not find the best model checkpoint.")
