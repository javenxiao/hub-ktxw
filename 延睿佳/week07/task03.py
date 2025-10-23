import json
import torch
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


# 数据加载和预处理 - 使用CMRC2018数据集
def load_cmrc2018_data(data_type='train', max_samples=1000):
    """加载CMRC2018中文阅读理解数据集"""

    if data_type == 'train':
        data_file = './cmrc2018_public/train.json'
    else:
        data_file = './cmrc2018_public/dev.json'

    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取问答对
    qa_pairs = []

    for article in data['data'][:max_samples]:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if qa['answers']:
                    answer = qa['answers'][0]['text']
                else:
                    answer = "未知"  # 处理没有答案的情况

                qa_pairs.append({
                    "instruction": f"根据以下上下文回答问题：{question}",
                    "input": context,
                    "output": answer,
                    "context": context,
                    "question": question,
                    "answer": answer
                })

    return qa_pairs


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
    将指令、上下文和答案转换为模型训练格式
    """
    # 构建系统提示
    system_prompt = "你是一个知识问答助手，根据给定的上下文回答问题。请确保答案准确且基于上下文。"

    # 构建完整的输入文本
    input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n上下文：{example['input']}\n问题：{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize输入部分
    input_tokens = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )

    # Tokenize输出部分（答案）
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
        output_dir="./output_qwen_qa",
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
        warmup_steps=100,
    )


# 问答预测函数
def predict_answer(model, tokenizer, context, question, device='cuda'):
    """使用微调后的模型进行问答预测"""

    # 构建输入
    system_prompt = "你是一个知识问答助手，根据给定的上下文回答问题。请确保答案准确且基于上下文。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"上下文：{context}\n问题：{question}"}
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
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # 避免重复
        )

    # 解码结果
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return answer.strip()


# 评估函数
def evaluate_model(model, tokenizer, test_data, device='cuda', num_samples=10):
    """评估模型性能"""
    print("开始评估模型...")

    correct = 0
    total = min(num_samples, len(test_data))

    for i in range(total):
        sample = test_data[i]
        context = sample['context']
        question = sample['question']
        expected_answer = sample['answer']

        try:
            predicted_answer = predict_answer(model, tokenizer, context, question, device)

            # 简单的答案匹配评估（实际应用中可以使用更复杂的评估指标）
            is_correct = expected_answer in predicted_answer or predicted_answer in expected_answer

            if is_correct:
                correct += 1

            print(f"样本 {i + 1}:")
            print(f"问题: {question}")
            print(f"预期答案: {expected_answer}")
            print(f"预测答案: {predicted_answer}")
            print(f"是否正确: {'是' if is_correct else '否'}")
            print("-" * 80)

        except Exception as e:
            print(f"处理样本 {i + 1} 时出错: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"评估准确率: {accuracy:.2f} ({correct}/{total})")

    return accuracy


# 主训练函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载CMRC2018数据集...")
    train_data = load_cmrc2018_data('train', max_samples=800)
    val_data = load_cmrc2018_data('dev', max_samples=200)

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
    tokenizer.save_pretrained("./output_qwen_qa")

    return model, tokenizer, val_data


# 测试函数
def test_qa_model(model, tokenizer, test_data=None, device='cuda'):
    """测试训练好的问答模型"""

    if test_data is None:
        # 如果没有提供测试数据，创建一些示例
        test_examples = [
            {
                "context": "秦始皇（前259年—前210年），嬴姓，赵氏，名政，是中国历史上第一个使用\"皇帝\"称号的君主。",
                "question": "秦始皇的原名是什么？",
                "answer": "嬴政"
            },
            {
                "context": "清华大学位于北京市海淀区，是中国著名的高等学府，成立于1911年。",
                "question": "清华大学在哪个城市？",
                "answer": "北京"
            },
            {
                "context": "《红楼梦》是中国古典四大名著之一，作者是曹雪芹，小说以贾、史、王、薛四大家族的兴衰为背景。",
                "question": "《红楼梦》的作者是谁？",
                "answer": "曹雪芹"
            }
        ]
    else:
        test_examples = test_data[:5]  # 使用前5个测试样本

    print("开始测试问答模型...")

    for i, example in enumerate(test_examples):
        try:
            context = example['context']
            question = example['question']
            expected_answer = example['answer']

            predicted_answer = predict_answer(model, tokenizer, context, question, device)

            print(f"测试样本 {i + 1}:")
            print(f"上下文: {context}")
            print(f"问题: {question}")
            print(f"预期答案: {expected_answer}")
            print(f"预测答案: {predicted_answer}")

            # 简单匹配检查
            match = expected_answer in predicted_answer or predicted_answer in expected_answer
            print(f"答案匹配: {'✓' if match else '✗'}")
            print("-" * 80)

        except Exception as e:
            print(f"处理测试样本 {i + 1} 时出错: {e}")
            print("-" * 80)


# 批量问答函数
def batch_qa_predict(model, tokenizer, contexts, questions, device='cuda'):
    """批量问答预测"""
    answers = []

    for context, question in tqdm(zip(contexts, questions), desc="批量问答", total=len(contexts)):
        try:
            answer = predict_answer(model, tokenizer, context, question, device)
            answers.append(answer)
        except Exception as e:
            print(f"处理问题 '{question}' 时出错: {e}")
            answers.append("")  # 出错时返回空字符串

    return answers


if __name__ == "__main__":
    # 执行训练
    print("开始训练Qwen-LoRA知识问答模型...")
    trained_model, trained_tokenizer, val_data = main()

    # 评估模型
    print("\n开始评估模型性能...")
    evaluate_model(trained_model, trained_tokenizer, val_data)

    # 测试模型
    print("\n开始测试模型...")
    test_qa_model(trained_model, trained_tokenizer, val_data)

    # 示例：如何使用训练好的模型进行预测
    print("\n示例预测:")
    example_context = "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。"
    example_question = "人工智能是什么？"

    answer = predict_answer(trained_model, trained_tokenizer, example_context, example_question)
    print(f"问题: {example_question}")
    print(f"答案: {answer}")
