import json

import torch
from datasets import Dataset
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer, DataCollatorForSeq2Seq,
)


# 数据加载
def load_and_preprocess_data():
    # 加载数据
    train = json.load(open('./cmrc2018_public/train.json', encoding="utf-8"))
    dev = json.load(open('./cmrc2018_public/dev.json', encoding="utf-8"))
    # 准备训练和验证数据
    train_paragraphs, train_questions, train_answers = prepare_dataset(train)
    val_paragraphs, val_questions, val_answers = prepare_dataset(dev)
    # 数据预处理
    train_dataset = Dataset.from_dict({
        'context': train_paragraphs[:],
        'question': train_questions[:],
        'answers': train_answers[:]
    })
    val_dataset = Dataset.from_dict({
        'context': val_paragraphs[:],
        'question': val_questions[:],
        'answers': val_answers[:]
    })

    return train_dataset, val_dataset


# 准备训练数据
def prepare_dataset(data):
    paragraphs = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            paragraphs.append(context)
            questions.append(qa['question'])
            answers.append({
                'answer_start': [qa['answers'][0]['answer_start']],
                'text': [qa['answers'][0]['text']]
            })

    return paragraphs, questions, answers


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
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n现在进行知识问答任务，根据给定的文本及问题进行回答<|im_end|>\n<|im_start|>user\n{'文本：' + example['context'] + '\n问题：' + example['question']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['answers']}", add_special_tokens=False)

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
        output_dir="./qa-qwen-lora-model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
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
    """预测单个知识问答"""
    messages = [
        {"role": "system",
         "content": "现在进行知识问答任务，根据给定的文本及问题进行回答"},
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
    train_ds, val_ds = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    # 4. 划分训练集和验证集
    train_ds = Dataset.from_pandas(train_ds.to_pandas().iloc[:1000])
    eval_ds = Dataset.from_pandas(val_ds.to_pandas().iloc[:100])

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
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
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
    model.load_adapter("./qa-qwen-lora-model/")
    model.cpu()

    # 测试预测
    test_text = "文本：《长眠不醒》（The Big Sleep）是雷蒙·钱德勒1939年的小说，曾二度（1946年、1978年）被改编为电影。这是首本以菲利普·马罗（Philip Marlowe）为主角的小说，并被认为是钱德勒最杰出的作品之一，并被认为是冷硬派（hard boiled）的侦探小说代表作品。故事剧情错综复杂，许多角色在不同的故事里皆有联系。2005年，它被时代杂志评为百大英文小说之列。《夜长梦多》（The Big Sleep），由霍华德·霍克斯（Howard Hawks）导演，亨弗莱·鲍嘉（Humphrey Bogart）及洛琳·白考儿（Lauren Bacall）主演Robert Mitchum主演" + "\n" + "问题：《长眠不醒》的作者是谁？"
    result = predict_entity(model, tokenizer, test_text)
    print(f"输入的文本及问题: {test_text}")
    print(f"预测的回答: {result}")


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    # single_example()
