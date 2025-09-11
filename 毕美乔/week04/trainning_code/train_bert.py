# train_bert_cls.py
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from utils import save_training_logs

# --------- 配置 ---------
DATA_PATH = "/home/bmq/project/bd/Week04/dataset/waimai_10k.csv"
MODEL_NAME = "/home/bmq/.cache/huggingface"   # 如需换模型，修改这里

OUTPUT_DIR = "../weights/bert_waimai_ckpt"
RANDOM_SEED = 42
NUM_LABELS = None  # 程序会根据数据自动推断
BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-5
MAX_LENGTH = 128
# ------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    set_seed(RANDOM_SEED)

    # 1. 读取数据
    df = pd.read_csv(DATA_PATH)
    # 期望 csv 有两列：'review' 和 'label'
    if "review" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'review' and 'label' columns.")

    # 自动推断标签数（分类类别）
    global NUM_LABELS
    label_values = sorted(df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(label_values)}
    id2label = {i: str(l) for l, i in label2id.items()}
    NUM_LABELS = len(label_values)
    df["label_id"] = df["label"].map(label2id)

    # 2. 划分训练/验证集（与之前一致使用 stratify）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["review"].values,
        df["label_id"].values,
        test_size=0.2,
        stratify=df["label_id"].values,
        random_state=RANDOM_SEED
    )

    train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
    val_df = pd.DataFrame({"text": val_texts, "label": val_labels})

    # 3. 加载分词器与模型
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )

    # 4. 构建 datasets.Dataset
    ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_df.reset_index(drop=True))

    # 5. Tokenize 函数（不需要 jieba，直接交给 tokenizer）
    def preprocess_fn(example):
        out = tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH)
        return out

    ds_train = ds_train.map(preprocess_fn, batched=False)
    ds_val = ds_val.map(preprocess_fn, batched=False)

    # 指定哪些 column 会作为模型输入
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c not in ("input_ids", "attention_mask", "label")])
    ds_val = ds_val.remove_columns([c for c in ds_val.column_names if c not in ("input_ids", "attention_mask", "label")])

    # 6. Data collator（自动 padding）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. TrainingArguments & Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        push_to_hub=False,
        seed=RANDOM_SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 8. 训练
    trainer.train()
    save_training_logs(trainer)
    # 9. 评估
    eval_result = trainer.evaluate()
    print("Evaluation:", eval_result)

    # 10. 保存模型与 tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")

    # 11. 预测示例
    sample_texts = ["味道很好，会再来", "太难吃了，再也不来了"]
    enc = tokenizer(sample_texts, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", padding=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for k, v in enc.items():
            enc[k] = v.to(device)
        outputs = model(**enc)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    # 把 label id 转回原始 label
    preds_labels = [id2label[int(p)] for p in preds]
    print("Sample predictions:", list(zip(sample_texts, preds_labels)))


if __name__ == "__main__":
    main()
