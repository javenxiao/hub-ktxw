import codecs
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification, AutoTokenizer, AutoModelForTokenClassification
)

from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: tag for i, tag in enumerate(tag_type)}
label2id = {tag: i for i, tag in enumerate(tag_type)}

train_lines = codecs.open('../dataset/msra/train/sentences.txt').readlines()[:1000]
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags = codecs.open('../dataset/msra/train/tags.txt').readlines()[:1000]
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags = [[label2id[x] for x in tag] for tag in train_tags]

val_lines = codecs.open('../dataset/msra/val/sentences.txt').readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('../dataset/msra/val/tags.txt').readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags = [[label2id[x] for x in tag] for tag in val_tags]

tokenizer = AutoTokenizer.from_pretrained('../models/Qwen/Qwen3-0.6B/')


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,
        max_length=128,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def prepare_dataset(texts, tags):
    tokens = [list(text) for text in texts]

    dataset = Dataset.from_dict({
        "tokens": tokens,
        "labels": tags,
    })
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset


train_dataset = prepare_dataset(train_lines, train_tags)
eval_dataset = prepare_dataset(val_lines, val_tags)

model = AutoModelForTokenClassification.from_pretrained(
    '../models/Qwen/Qwen3-0.6B/',
    num_labels=len(tag_type),
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True
)

config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

model.to(device)

training_args = TrainingArguments(
    output_dir='./qwen-lora-ner',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    do_eval=True,
    eval_steps=50,
    num_train_epochs=20,
    save_steps=50,
    learning_rate=3e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    report_to="none",
    no_cuda=(device.type != 'cuda')
)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    accuracy = accuracy_score(flat_true_labels, flat_true_predictions)

    report = classification_report(
        flat_true_labels,
        flat_true_predictions,
        output_dict=True,
        zero_division=0
    )

    f1_scores = {}
    for label in tag_type:
        if label in report:
            f1_scores[f"{label}_f1"] = report[label]["f1-score"]

    return {
        "accuracy": accuracy,
        **f1_scores
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("train")
trainer.train()

trainer.save_model()
tokenizer.save_pretrained("./ner-qwen-models")

print("eval")
eval_results = trainer.evaluate()
print(f"eval results: {eval_results}")

test_sentences = [
    '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',  # 人、位置
    '人工智能是未来的希望，也是中国和美国的冲突点。',
    '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
    '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
    '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
    '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
]


def predict_ner(sentence):
    model.to('cpu')
    tokens = list(sentence)
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    print(f"predictions: {predictions}")

    predicted_labels = [id2label[p.item()] for p in predictions[0]]

    word_ids = inputs.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:
            aligned_labels.append(predicted_labels[i])
        previous_word_idx = word_idx

    if len(aligned_labels) > len(tokens):
        aligned_labels = aligned_labels[:len(tokens)]
    elif len(aligned_labels) < len(tokens):
        aligned_labels.extend(['0'] * (len(tokens) - len(aligned_labels)))

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

    model.to(device)

    return entities
