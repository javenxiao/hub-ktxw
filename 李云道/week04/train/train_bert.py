import os.path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

'''
使用bert模型微调
'''


def load_data(file_path: str) -> (list, list):
    df = pd.read_csv(file_path, sep=",", header=0)
    # print(df.head(3))
    labels = df["label"].tolist()
    texts = df["review"].tolist()
    return texts, labels


def evaluator(input: tuple) -> Dict:
    predict, target = input
    predict = np.argmax(predict, axis=1)
    return {"correct": (target == predict).mean()}


def get_trainer(config: Dict, texts: list, labels: list) -> Trainer:
    # 划分训练集和测试集
    x_train, x_test, train_labels, test_labels = train_test_split(
        texts,  # 文本数据
        labels,  # 对应的数字标签
        test_size=config["test_size"],  # 测试集比例为20%
        stratify=labels  # 确保训练集和测试集的标签分布一致
    )

    # 从bert模型加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    model = BertForSequenceClassification.from_pretrained(
        config["bert_path"],
        num_labels=config["num_labels"]  # 判别式模型，分多少类
    )

    x_train = tokenizer(x_train, truncation=True, padding=True, max_length=config["max_length"])
    x_test = tokenizer(x_test, truncation=True, padding=True, max_length=config["max_length"])

    # 转换为torch的dataset
    train_dataset = Dataset.from_dict({
        'input_ids': x_train['input_ids'],  # 文本的token ID
        'attention_mask': x_train['attention_mask'],  # 注意力掩码
        'labels': train_labels  # 对应的标签
    })
    test_dataset = Dataset.from_dict({
        'input_ids': x_test['input_ids'],
        'attention_mask': x_test['attention_mask'],
        'labels': test_labels
    })

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=config["model_path"],  # 训练输出目录，用于保存模型和状态
        num_train_epochs=config["num_epochs"],  # 训练的总轮数
        per_device_train_batch_size=config["batch_size"],  # 训练时每个设备（GPU/CPU）的批次大小
        per_device_eval_batch_size=config["batch_size"],  # 评估时每个设备的批次大小
        warmup_steps=500,  # 学习率预热的步数，有助于稳定训练
        weight_decay=0.01,  # 权重衰减，用于防止过拟合
        logging_dir='./logs',  # 日志存储目录
        logging_steps=100,  # 每隔100步记录一次日志
        eval_strategy="epoch",  # 每训练完一个 epoch 进行一次评估
        save_strategy="epoch",  # 每训练完一个 epoch 保存一次模型
        load_best_model_at_end=True,  # 训练结束后加载效果最好的模型
    )

    trainer = Trainer(
        model=model,  # 要训练的模型
        args=training_args,  # 训练参数
        train_dataset=train_dataset,  # 训练数据集
        eval_dataset=test_dataset,  # 评估数据集
        compute_metrics=evaluator,  # 用于计算评估指标的函数
    )
    return trainer


def main(config: Dict):
    # 读取数据
    texts, labels = load_data(config["data_path"])
    config["num_labels"] = len(set(labels))

    trainer = get_trainer(config, texts, labels)

    # 训练模型
    trainer.train()
    # 评估模型
    trainer.evaluate()

    best_model_path = trainer.state.best_model_checkpoint
    model_path = [config["model_path"], config["model_name"]]
    model_path = os.path.join(*model_path).__str__()

    if best_model_path:
        best_model = BertForSequenceClassification.from_pretrained(best_model_path)
        print("best model path:", best_model_path)
        torch.save(best_model.state_dict(), model_path)
    else:
        print("No best model found.")


if __name__ == '__main__':
    from config import Config

    # 让路径指向上一级目录
    Config["data_path"] = "../" + Config["data_path"]
    Config["model_path"] = "../" + Config["model_path"]

    main(Config)
