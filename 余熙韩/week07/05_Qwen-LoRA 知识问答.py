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
from peft import LoraConfig, TaskType, get_peft_model
import torch
from tqdm import tqdm

# 加载CMRC2018数据集
def load_cmrc2018_data(train_file_path, dev_file_path):
    """加载CMRC2018数据集并转换为训练所需格式"""
    # 加载JSON数据
    train_data = json.load(open(train_file_path, 'r', encoding='utf-8'))
    dev_data = json.load(open(dev_file_path, 'r', encoding='utf-8'))
    
    # 准备训练数据
    def prepare_dataset(data, max_samples=None):
        instructions = []
        inputs = []
        outputs = []
        
        count = 0
        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text']
                    
                    # 构建指令格式
                    instruction = "请根据以下上下文回答问题。"
                    user_input = f"上下文：{context}\n问题：{question}"
                    
                    instructions.append(instruction)
                    inputs.append(user_input)
                    outputs.append(answer)
                    
                    count += 1
                    if max_samples and count >= max_samples:
                        return instructions, inputs, outputs
        
        return instructions, inputs, outputs
    
    # 处理训练数据和验证数据
    train_instructions, train_inputs, train_outputs = prepare_dataset(train_data, max_samples=1000)
    val_instructions, val_inputs, val_outputs = prepare_dataset(dev_data, max_samples=100)
    
    # 创建DataFrame
    train_df = pd.DataFrame({
        'instruction': train_instructions,
        'input': train_inputs,
        'output': train_outputs
    })
    
    val_df = pd.DataFrame({
        'instruction': val_instructions,
        'input': val_inputs,
        'output': val_outputs
    })
    
    # 转换为Hugging Face Dataset
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    return train_ds, val_ds

# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化Qwen3-0.6B模型和tokenizer"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # 检查设备兼容性（MPS设备处理）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 尝试加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if device.type != "mps" else torch.float32  # MPS不支持float16
        )
        
        # 如果是MPS设备，手动移至MPS
        if device.type == "mps":
            model = model.to(device)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("尝试使用CPU加载模型...")
        # 退回到CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32
        )
    
    return tokenizer, model

# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """处理单个样本，转换为模型训练格式"""
    # 构建ChatML格式的指令和响应
    system_message = "你是一个知识问答助手，根据提供的上下文准确回答用户的问题。"
    user_message = f"{example['instruction']}\n{example['input']}"
    assistant_message = example['output']
    
    # 构建完整的输入文本
    input_text = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize输入和输出
    inputs = tokenizer(input_text, add_special_tokens=False)
    outputs = tokenizer(assistant_message, add_special_tokens=False)
    
    # 组合输入ID和注意力掩码
    input_ids = inputs["input_ids"] + outputs["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"] + outputs["attention_mask"] + [1]
    
    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(inputs["input_ids"]) + outputs["input_ids"] + [tokenizer.pad_token_id]
    
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

# 设置训练参数
def setup_training_args(output_dir="./output_Qwen3_qa"):
    """设置训练参数"""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        save_steps=100,
        logging_steps=100,
        eval_steps=100,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="none"
    )

# 预测函数
def predict_answer(model, tokenizer, context, question, device='cpu'):
    """预测给定上下文和问题的答案"""
    # 构建ChatML格式的消息
    messages = [
        {"role": "system", "content": "你是一个知识问答助手，根据提供的上下文准确回答用户的问题。"},
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
            max_new_tokens=128,
            do_sample=False,  # 确定性生成
            temperature=0.0,  # 温度设为0以获得最确定的输出
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 清理答案，移除不必要的前缀
    answer = response.strip()
    if answer.startswith("答案："):
        answer = answer[3:]
    
    return answer

# 批量预测
def batch_predict(model, tokenizer, contexts, questions, device='cpu'):
    """批量预测测试集的答案"""
    pred_answers = []
    
    for context, question in tqdm(zip(contexts, questions), desc="预测答案", total=len(contexts)):
        try:
            pred_answer = predict_answer(model, tokenizer, context, question, device)
            pred_answers.append(pred_answer)
        except Exception as e:
            print(f"预测时出错: {e}")
            pred_answers.append("")
    
    return pred_answers

# 主函数
def main():
    """主执行函数"""
    # 设置路径
    train_file_path = './cmrc2018_public/train.json'
    dev_file_path = './cmrc2018_public/dev.json'
    model_path = '../models/Qwen/Qwen3-0.6B'
    
    # 1. 加载和处理数据
    print("加载CMRC2018数据集...")
    train_ds, val_ds = load_cmrc2018_data(train_file_path, dev_file_path)
    
    # 保存验证集的原始上下文和问题，用于评估
    val_contexts = []
    val_questions = []
    val_answers = []
    for example in val_ds:
        # 从input中提取上下文和问题
        input_text = example['input']
        if '上下文：' in input_text and '\n问题：' in input_text:
            context_part, question_part = input_text.split('\n问题：', 1)
            context = context_part.replace('上下文：', '').strip()
            question = question_part.strip()
            val_contexts.append(context)
            val_questions.append(question)
            val_answers.append(example['output'])
    
    # 2. 初始化模型和tokenizer
    print("初始化Qwen3-0.6B模型和tokenizer...")
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    
    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(process_func_with_tokenizer, remove_columns=val_ds.column_names)
    
    # 4. 设置LoRA
    print("配置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)
    
    # 5. 设置训练参数
    print("设置训练参数...")
    training_args = setup_training_args()
    
    # 6. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )
    
    trainer.train()
    
    # 7. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 8. 在验证集上评估
    print("在验证集上测试...")
    # 获取设备
    device = next(model.parameters()).device
    
    # 预测前几个样本
    for i in range(min(3, len(val_contexts))):
        context = val_contexts[i]
        question = val_questions[i]
        expected_answer = val_answers[i]
        
        predicted_answer = predict_answer(model, tokenizer, context, question, device)
        
        print(f"问题 {i+1}: {question}")
        print(f"预期答案: {expected_answer}")
        print(f"预测答案: {predicted_answer}")
        print(f"匹配度: {expected_answer in predicted_answer or predicted_answer in expected_answer}")
        print()

# 单独测试函数
def test_single_example():
    """测试单个问答示例"""
    model_path = '../models/Qwen/Qwen3-0.6B'
    
    # 初始化模型和tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    
    # 加载训练好的LoRA权重
    try:
        model.load_adapter("./output_Qwen3_qa/")
    except Exception as e:
        print(f"加载LoRA权重失败: {e}")
        print("使用原始模型进行测试...")
    
    # 移至CPU以兼容所有环境
    model.cpu()
    
    # 测试示例
    context = "《哈利·波特》是英国作家J·K·罗琳于1997～2007年所著的魔幻文学系列小说，共7部。其中前六部以霍格沃茨魔法学校为主要舞台，描写的是主人公——年轻的巫师学生哈利·波特在霍格沃茨前后六年的学习生活和冒险故事；第七本描写的是哈利·波特在第二次魔法界大战中在外寻找魂器并消灭伏地魔的故事。"
    question = "《哈利·波特》系列小说共有多少部？"
    
    # 预测答案
    answer = predict_answer(model, tokenizer, context, question, device='cpu')
    
    print(f"上下文: {context}")
    print(f"问题: {question}")
    print(f"预测答案: {answer}")

if __name__ == "__main__":
    # 执行主函数进行训练
    main()
    
    # 或者单独测试
    # test_single_example()