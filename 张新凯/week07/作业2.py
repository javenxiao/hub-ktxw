import codecs

import torch
from datasets import Dataset
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)


# 数据加载
def load_data():
    # 加载训练数据
    train_sentences = codecs.open('./msra/train/sentences.txt', encoding="utf-8").readlines()[:]
    train_sentences = [x.replace(' ', '').strip() for x in train_sentences]
    train_tags = codecs.open('./msra/train/tags.txt', encoding="utf-8").readlines()[:]

    # 加载验证数据
    val_sentences = codecs.open('./msra/val/sentences.txt', encoding="utf-8").readlines()[:]
    val_sentences = [x.replace(' ', '').strip() for x in val_sentences]
    val_tags = codecs.open('./msra/val/tags.txt', encoding="utf-8").readlines()[:]

    return train_sentences, train_tags, val_sentences, val_tags


# 数据预处理
def preprocess(sentences, tags):
    dataset = Dataset.from_dict({
        "sentences": sentences,
        "tags": tags
    })
    return dataset


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        # dtype=torch.float16  # 使用半精度(float16)可以减少内存占用
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=1024):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n现在进行实体标注任务，实体包括组织（ORG）、人名（PER）和地点（LOC），对文本中的每个字符标注BIO标签，标签包括['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']。输出n个标签，与文本中的n个字符一一对应，格式为：标签1 标签2 …… 标签n。注意除了按格式输出n个标签外不要输出包括思考过程或者结果解释等任何其他内容。<|im_end|>\n<|im_start|>user\n{example['sentences']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['tags']}", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

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
        output_dir="./ner-qwen-lora-model",
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=2,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=False,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_entity(model, tokenizer, text, device='cpu'):
    text = ' '.join(list(text))
    """预测单个文本的实体标注"""
    messages = [
        {"role": "system",
         "content": "现在进行实体标注任务，实体包括组织（ORG）、人名（PER）和地点（LOC），对文本中的每个字符标注BIO标签，标签包括['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']。输出n个标签，与文本中的n个字符一一对应，格式为：标签1 标签2 …… 标签n。注意除了按格式输出n个标签外不要输出包括思考过程或者结果解释等任何其他内容。"},
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
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_sentences, train_tags, val_sentences, val_tags = load_data()
    train_ds = preprocess(train_sentences, train_tags)
    val_ds = preprocess(val_sentences, val_tags)

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    # 4. 划分训练集和验证集
    train_ds = Dataset.from_pandas(train_ds.to_pandas().iloc[:4000])
    eval_ds = Dataset.from_pandas(val_ds.to_pandas().iloc[:300])

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

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
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True
        ),
    )

    trainer.train()

    # 8. 保存模型
    # print("保存模型...")
    # trainer.save_model()
    # tokenizer.save_pretrained("./output_Qwen")


# 单独测试函数
def single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "../models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./ner-qwen-lora-model/")
    model.cpu()

    # 测试预测
    test_text = "明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。"
    result = predict_entity(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"实体标注: {result}")


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    # single_example()
