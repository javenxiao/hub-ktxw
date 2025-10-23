import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch
import codecs
import json

# 定义实体类型（与04_BERT实体抽取.py保持一致）
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


# 数据加载和预处理 - 使用MSRA数据集
def load_msra_data(data_type='train', max_samples=1000):
    """加载MSRA命名实体识别数据集"""

    if data_type == 'train':
        sentences_file = './msra/train/sentences.txt'
        tags_file = './msra/train/tags.txt'
    else:
        sentences_file = './msra/val/sentences.txt'
        tags_file = './msra/val/tags.txt'

    # 加载句子
    with codecs.open(sentences_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:max_samples]
    lines = [x.replace(' ', '').strip() for x in lines]

    # 加载标签
    with codecs.open(tags_file, 'r', encoding='utf-8') as f:
        tags = f.readlines()[:max_samples]
    tags = [x.strip().split(' ') for x in tags]

    # 转换为实体抽取格式
    data = []
    for sentence, tag_sequence in zip(lines, tags):
        # 提取实体
        entities = extract_entities_from_bio(sentence, tag_sequence)

        # 构建指令格式
        instruction = f"从以下文本中提取实体：{sentence}"

        # 构建输出格式：实体类型: 实体名称
        if entities:
            output = "；".join([f"{entity_type}: {entity}" for entity, entity_type in entities])
        else:
            output = "未识别到实体"

        data.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "sentence": sentence,
            "entities": entities
        })

    return data


def extract_entities_from_bio(sentence, tag_sequence):
    """从BIO标签序列中提取实体"""
    entities = []
    current_entity = ""
    current_type = ""

    for char, tag in zip(sentence, tag_sequence):
        if tag.startswith('B-'):
            # 开始新实体
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = char
            current_type = tag[2:]
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
            # 继续当前实体
            current_entity += char
        else:
            # 实体结束或遇到O标签
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""

    # 处理最后一个实体
    if current_entity:
        entities.append((current_entity, current_type))

    return entities


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建完整的输入文本
    input_text = f"<|im_start|>system\n你是一个实体识别专家，需要从文本中提取人名(PER)、地名(LOC)和组织名(ORG)。<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize输入部分
    input_tokens = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )

    # Tokenize输出部分
    output_tokens = tokenizer(
        example['output'],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )

    # 组合输入和输出
    input_ids = input_tokens['input_ids'] + output_tokens['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = input_tokens['attention_mask'] + output_tokens['attention_mask'] + [1]

    # 构建标签（输入部分用-100忽略，只计算输出部分的损失）
    labels = [-100] * len(input_tokens['input_ids']) + output_tokens['input_ids'] + [tokenizer.pad_token_id]

    # 截断到最大长度
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        output_dir="./output_qwen_ner",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        logging_steps=50,
        eval_steps=100,
        num_train_epochs=5,
        save_steps=200,
        learning_rate=2e-4,
        gradient_checkpointing=True,
        report_to="none",
        save_total_limit=3,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


# 实体识别预测函数
def predict_entities(model, tokenizer, text, device='cuda'):
    """使用微调后的模型进行实体识别"""

    # 构建输入
    instruction = f"从以下文本中提取实体：{text}"
    messages = [
        {"role": "system", "content": "你是一个实体识别专家，需要从文本中提取人名(PER)、地名(LOC)和组织名(ORG)。"},
        {"role": "user", "content": instruction}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=False,  # 使用贪婪解码确保结果稳定
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 解码结果
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 解析实体结果
    entities = parse_entities_from_response(response)

    return entities, response


def parse_entities_from_response(response):
    """从模型响应中解析实体"""
    entities = []

    # 按分号分割不同的实体
    entity_parts = response.split('；')

    for part in entity_parts:
        part = part.strip()
        if ':' in part:
            # 分割类型和实体名称
            type_entity = part.split(':', 1)
            if len(type_entity) == 2:
                entity_type = type_entity[0].strip()
                entity_name = type_entity[1].strip()

                # 验证实体类型是否有效
                if entity_type in ['PER', 'LOC', 'ORG']:
                    entities.append((entity_name, entity_type))

    return entities


# 主训练函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载MSRA数据集...")
    train_data = load_msra_data('train', max_samples=800)
    val_data = load_msra_data('val', max_samples=200)

    # 转换为DataFrame和Dataset
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"  # 使用较小模型节省资源
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    tokenized_train_ds = train_ds.map(
        process_func_with_tokenizer,
        remove_columns=train_ds.column_names
    )
    tokenized_val_ds = val_ds.map(
        process_func_with_tokenizer,
        remove_columns=val_ds.column_names
    )

    # 4. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 5. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 6. 创建数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )

    # 7. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_qwen_ner")

    return model, tokenizer


# 测试函数
def test_model(model, tokenizer, test_sentences=None):
    """测试训练好的模型"""

    if test_sentences is None:
        test_sentences = [
            '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
            '人工智能是未来的希望，也是中国和美国的冲突点。',
            '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
            '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
            '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
            '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
        ]

    device = next(model.parameters()).device

    print("开始测试模型...")
    for sentence in test_sentences:
        try:
            entities, raw_response = predict_entities(model, tokenizer, sentence, device)
            print(f"句子: {sentence}")
            print(f"模型响应: {raw_response}")
            if entities:
                for entity, entity_type in entities:
                    print(f"  {entity_type}: {entity}")
            else:
                print("  未识别到实体")
            print("-" * 50)
        except Exception as e:
            print(f"处理句子时出错: {sentence}")
            print(f"错误信息: {e}")
            print("-" * 50)


if __name__ == "__main__":
    # 执行训练
    print("开始训练Qwen-LoRA实体识别模型...")
    trained_model, trained_tokenizer = main()

    # 测试模型
    print("\n开始测试训练好的模型...")
    test_model(trained_model, trained_tokenizer)
