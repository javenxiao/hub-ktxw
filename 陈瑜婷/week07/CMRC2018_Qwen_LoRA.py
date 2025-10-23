import json
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


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理CMRC2018数据"""
    # 读取训练数据
    with open('cmrc2018_public/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 提取问答对
    qa_pairs = []
    for article in train_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # 取第一个答案作为标准答案
                answer = qa['answers'][0]['text'] if qa['answers'] else ""
                
                qa_pairs.append({
                    'instruction': question,
                    'output': answer,
                    'input': context
                })
    
    # 转换为DataFrame
    df = pd.DataFrame(qa_pairs)
    
    # 转换为Hugging Face Dataset
    ds = Dataset.from_pandas(df)
    
    return ds


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
        torch_dtype=torch.float16  # 使用半精度减少内存占用
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    将问题、上下文和答案转换为模型训练格式
    """
    # 构建指令部分 - 知识问答任务
    instruction_text = f"<|im_start|>system\n现在进行知识问答任务，请根据给定的上下文回答问题<|im_end|>\n<|im_start|>user\n上下文：{example['input']}\n问题：{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

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
        output_dir="./output_CMRC2018_Qwen",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=3,  # 减少训练轮数
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",  # 禁用wandb等报告工具
        max_grad_norm=1.0,  # 添加梯度裁剪
        warmup_steps=100,   # 添加预热步数
    )


# 预测函数
def predict_answer(model, tokenizer, context, question, device='cpu'):
    """预测单个问题的答案"""
    messages = [
        {"role": "system", "content": "现在进行知识问答任务，请根据给定的上下文回答问题"},
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
            model_inputs.input_ids,
            max_new_tokens=128,  # 减少生成长度，因为答案通常较短
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
def batch_predict(model, tokenizer, test_data, device='cuda'):
    """批量预测测试集的答案"""
    pred_answers = []

    for item in tqdm(test_data, desc="预测答案"):
        try:
            pred_answer = predict_answer(
                model, tokenizer, 
                item['input'], item['instruction'], 
                device
            )
            pred_answers.append(pred_answer)
        except Exception as e:
            print(f"预测问题时出错: {e}")
            pred_answers.append("")  # 出错时添加空字符串

    return pred_answers


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载CMRC2018数据...")
    ds = load_and_preprocess_data()
    print(f"总共加载了 {len(ds)} 个问答对")

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

    # 4. 划分训练集和验证集
    # 使用前80%作为训练集，后20%作为验证集
    train_size = int(0.8 * len(ds))
    train_ds = Dataset.from_pandas(ds.to_pandas().iloc[:train_size])
    eval_ds = Dataset.from_pandas(ds.to_pandas().iloc[train_size:])

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

    print(f"训练集大小: {len(train_tokenized)}")
    print(f"验证集大小: {len(eval_tokenized)}")

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
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_CMRC2018_Qwen")


# 单独测试函数
def test_single_example():
    """测试单个问答示例"""
    model_path = "E:/LearnAI/BadouClass/models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_CMRC2018_Qwen/")
    model.cpu()

    # 测试预测
    test_context = "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。"
    test_question = "范廷颂是什么时候被任为主教的？"
    
    result = predict_answer(model, tokenizer, test_context, test_question)
    print(f"上下文: {test_context}")
    print(f"问题: {test_question}")
    print(f"预测答案: {result}")


if __name__ == "__main__":
    # 执行主函数
    main()

    # 单独测试
    test_single_example()
