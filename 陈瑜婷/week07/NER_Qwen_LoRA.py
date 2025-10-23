import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch
import os
from sklearn.metrics import classification_report, f1_score
import numpy as np


# 数据加载和预处理
def load_msra_data():
    """加载MSRA数据集"""
    def read_conll_file(sentences_file, tags_file):
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        with open(tags_file, 'r', encoding='utf-8') as f:
            tags = f.readlines()
        
        data = []
        for sent, tag in zip(sentences, tags):
            # 去除换行符并分割
            tokens = sent.strip().split()
            labels = tag.strip().split()
            
            # 确保token和label数量一致
            if len(tokens) == len(labels):
                data.append({
                    'tokens': tokens,
                    'labels': labels
                })
        
        return data
    
    # 加载训练、验证和测试数据
    train_data = read_conll_file('msra/train/sentences.txt', 'msra/train/tags.txt')
    val_data = read_conll_file('msra/val/sentences.txt', 'msra/val/tags.txt')
    test_data = read_conll_file('msra/test/sentences.txt', 'msra/test/tags.txt')
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    return train_data, val_data, test_data


# 获取标签映射
def get_label_mapping():
    """获取标签到ID的映射"""
    # 从tags.txt文件读取所有可能的标签
    with open('msra/tags.txt', 'r', encoding='utf-8') as f:
        unique_tags = [line.strip() for line in f.readlines()]
    
    # 创建标签映射
    label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2label = {idx: tag for tag, idx in label2id.items()}
    
    print(f"标签数量: {len(unique_tags)}")
    print(f"标签列表: {unique_tags}")
    
    return label2id, id2label


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16  # 使用半精度减少内存占用
    )

    return tokenizer, model


# 数据处理函数
def process_ner_data(example, tokenizer, label2id, max_length=512):
    """
    处理NER数据的函数
    将token序列和标签序列转换为模型训练格式
    """
    # 构建输入文本（将tokens连接成句子）
    input_text = " ".join(example['tokens'])
    
    # 构建标签文本（将labels连接成序列）
    label_text = " ".join(example['labels'])
    
    # 构建指令部分 - 使用ChatML格式
    instruction_text = f"<|im_start|>system\n现在进行中文命名实体识别任务，请识别文本中的人名(PER)、地名(LOC)、机构名(ORG)等实体。<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    
    # 构建响应部分
    response = tokenizer(f"{label_text}", add_special_tokens=False)
    
    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    
    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    
    # 截断超过最大长度的序列
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
    
    # 对什么模型，以什么方式进行微调
    config = LoraConfig(
        # 任务类型，自回归语言建模
        task_type=TaskType.CAUSAL_LM,
        
        # 对什么层的默写模块进行高效微调
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
        output_dir="./msra_ner_output",
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_ner(model, tokenizer, text, device='cpu'):
    """预测单个文本的NER标签"""
    messages = [
        {"role": "system", "content": "现在进行中文命名实体识别任务，请识别文本中的人名(PER)、地名(LOC)、机构名(ORG)等实体。"},
        {"role": "user", "content": text}
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
            model_inputs.input_ids,
            max_new_tokens=256,  # 减少生成长度
            do_sample=True,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


# 批量预测
def batch_predict_ner(model, tokenizer, test_data, device='cuda'):
    """批量预测测试集的NER标签"""
    pred_labels = []
    
    for example in tqdm(test_data, desc="预测NER标签"):
        try:
            # 构建输入文本
            input_text = " ".join(example['tokens'])
            pred_label = predict_ner(model, tokenizer, input_text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串
    
    return pred_labels



# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载MSRA数据...")
    train_data, val_data, test_data = load_msra_data()
    
    # 2. 获取标签映射
    print("获取标签映射...")
    label2id, id2label = get_label_mapping()
    
    # 3. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "E:/LearnAI/BadouClass/models/Qwen/Qwen3-0.6B"  # 根据实际路径调整
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    
    # 4. 处理数据
    print("处理训练数据...")
    def process_func_with_tokenizer(example):
        return process_ner_data(example, tokenizer, label2id)
    
    # 转换为Dataset格式
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # 处理数据
    train_tokenized = train_dataset.map(process_func_with_tokenizer, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(process_func_with_tokenizer, remove_columns=val_dataset.column_names)
    
    # 5. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)
    
    # 6. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()
    
    # 7. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )
    
    trainer.train()
    
    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./msra_ner_output")
    
    
    return model, tokenizer


# 单独测试函数
def test_single_example():
    """测试单个样本的预测"""
    model_path = "E:/LearnAI/BadouClass/models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    
    # 加载训练好的LoRA权重
    model.load_adapter("./msra_ner_output/")
    model.cpu()
    
    # 测试预测
    test_text = "中 国 人 民 大 学 位 于 北 京 市"
    result = predict_ner(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"预测标签: {result}")


if __name__ == "__main__":
    # 执行主函数
    model, tokenizer = main()
    
    # 单独测试
    test_single_example()
