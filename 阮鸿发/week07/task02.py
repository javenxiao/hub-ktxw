import codecs
import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
import warnings
from peft import LoraConfig, TaskType, get_peft_model



warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 加载训练数据
train_lines = codecs.open('./msra/train/sentences.txt',encoding='utf-8').readlines()[:1000]
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags = codecs.open('./msra/train/tags.txt',encoding='utf-8').readlines()[:1000]
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags = [[label2id[x] for x in tag] for tag in train_tags]

# 加载验证数据
val_lines = codecs.open('./msra/val/sentences.txt',encoding='utf-8').readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('./msra/val/tags.txt',encoding='utf-8').readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags = [[label2id[x] for x in tag] for tag in val_tags]

# 初始化tokenizer
tokenizer = BertTokenizerFast.from_pretrained('../../models/google-bert/bert-base-chinese/')


# 对数据进行tokenize
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,
        max_length=128,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # 特殊token设置为-100，在计算损失时会被忽略
            if word_idx is None:
                label_ids.append(-100)
            # 当前单词与之前单词相同（子词）
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # 当前单词是之前单词的一部分
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 准备数据集
def prepare_dataset(texts, tags):
    # 将文本拆分为字符列表
    tokens = [list(text) for text in texts]

    # 创建数据集
    dataset = Dataset.from_dict({
        "tokens": tokens,
        "labels": tags
    })

    # 对数据集进行tokenize
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


# 准备训练和验证数据集
train_dataset = prepare_dataset(train_lines, train_tags)
eval_dataset = prepare_dataset(val_lines, val_tags)

# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        target_modules=["query","key","value","dense"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./ner-bert-model",
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=False,
        report_to="none"  # 禁用wandb等报告工具
    )



# 加载模型
model = BertForTokenClassification.from_pretrained(
    '../../models/google-bert/bert-base-chinese/',
    num_labels=len(tag_type),
    id2label=id2label,
    label2id=label2id
)

# 将模型移动到相应设备
model.to(device)


# 设置训练参数
training_args = TrainingArguments(
    output_dir='./ner-bert-model',
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    # 添加设备相关设置
    no_cuda=(device.type != "cuda"),  # 如果不是CUDA设备，禁用CUDA
)

# 数据收集器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)


# 定义计算指标的函数
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除忽略的索引（-100）
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 计算准确率
    flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    accuracy = accuracy_score(flat_true_labels, flat_true_predictions)

    # 计算分类报告
    report = classification_report(
        flat_true_labels,
        flat_true_predictions,
        output_dict=True,
        zero_division=0
    )

    # 提取各个类别的F1分数
    f1_scores = {}
    for label in tag_type:
        if label in report:
            f1_scores[f"{label}_f1"] = report[label]["f1-score"]

    return {
        "accuracy": accuracy,
        **f1_scores
    }

# 5. 设置LoRA
print("设置LoRA...")
model.enable_input_require_grads()
model = setup_lora(model)

# 6. 配置训练参数
print("配置训练参数...")
training_args = setup_training_args()

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
print("开始训练...")
trainer.train()

# 保存模型
trainer.save_model()
tokenizer.save_pretrained('./ner-bert-model')

# 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 测试预测
test_sentences = [
    '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
    '人工智能是未来的希望，也是中国和美国的冲突点。',
    '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
    '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
    '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
    '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
]

# 可选：如果MPS设备仍有问题，可以强制使用CPU进行预测
def predict_cpu(sentence):
    # 临时将模型移动到CPU进行预测
    model.to('cpu')

    # 将句子转换为字符列表
    tokens = list(sentence)

    # Tokenize
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    print(predictions)

    # 将预测结果转换为标签
    predicted_labels = [id2label[p.item()] for p in predictions[0]]

    # 对齐标签
    word_ids = inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_labels.append(predicted_labels[i])
        previous_word_idx = word_idx

    # 确保标签数量与token数量一致
    if len(aligned_labels) > len(tokens):
        aligned_labels = aligned_labels[:len(tokens)]
    elif len(aligned_labels) < len(tokens):
        aligned_labels.extend(['O'] * (len(tokens) - len(aligned_labels)))

    # 提取实体
    entities = []
    current_entity = ""
    current_type = ""

    for token, label in zip(tokens, aligned_labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = token
            current_type = label[2:]
        elif label.startswith('I-') and current_entity and current_type == label[2:]:
            current_entity += token
        else:
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = ""
            current_type = ""
            if label.startswith('B-'):
                current_entity = token
                current_type = label[2:]

    if current_entity:
        entities.append((current_entity, current_type))

    # 将模型移回原设备
    model.to(device)

    return entities


# 如果MPS预测失败，使用CPU预测
for sentence in test_sentences:
    try:
        entities = predict_cpu(sentence)
        print(f"句子: {sentence}")
        if entities:
            for entity, entity_type in entities:
                print(f"  {entity_type}: {entity}")
        else:
            print("  未识别到实体")
        print()
    except Exception as e:
        print(f"处理句子时出错: {sentence}")
        print(f"错误信息: {e}")
        print()
