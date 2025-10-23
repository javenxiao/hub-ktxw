import json

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,  
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer, 
)
from datasets import Dataset 
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report 
import os
import warnings


def load_and_preprocess_data():
    # 加载数据
    train = json.load(open('./cmrc2018_public/train.json'))
    dev = json.load(open('./cmrc2018_public/dev.json'))


    train_all = []
    for each_item in train['data']:
        context = each_item['paragraphs'][0]['context']
        for qa in each_item['paragraphs'][0]['qas']:
            question = qa['question']
            answer = qa['answers'][0]['text']
            train_all.append({
                "instruction": f"基于这一段文本: {context}, 回答问题：{question}",
                "output": answer,
                "input": ""
            })

    dev_answers = []
    dev_paragraphs = []   # 用来存文章段落
    dev_questions = []    # 用来存问题
    dev_all = []
    for each_item in dev['data']:
        context = each_item['paragraphs'][0]['context']
        for qa in each_item['paragraphs'][0]['qas']:
            dev_paragraphs.append(context)
            question = qa['question']
            dev_questions.append(question)
            answer = qa['answers'][0]['text']
            dev_answers.append(answer)
            dev_all.append({
                "instruction": f"基于这一段文本: {context}, 回答问题：{question}",
                "output": answer,
                "input": ""
            })

    ds_train = Dataset.from_pandas(pd.DataFrame(train_all[: 200]))
    ds_dev = Dataset.from_pandas(pd.DataFrame(dev_all[: 200]))

    return ds_train, ds_dev, dev_paragraphs[: 200], dev_questions[: 200], dev_answers[: 200]


def initialize_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False, 
        trust_remote_code=True 
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,  # 不自动分配设备
        torch_dtype=torch.float16,  # 使用半精度减少内存占用
        trust_remote_code=True  # 信任远程代码
    )
    model.to("mps")

    return tokenizer, model


def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    instruction_text = f"<|im_start|>system\n现在进行知识问答，要求根据提供的段落回答问题，要求只给出答案即可。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)   # 分词处理


    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

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
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  
        r=8,   
        lora_alpha=32,  
        lora_dropout=0.1  
    )


    model = get_peft_model(model, config)
    # 打印可训练的参数数量
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./qa-qwen_lora-model",
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=4, 
        logging_steps=100,  
        do_eval=True, 
        eval_steps=50,   
        num_train_epochs=3, 
        save_steps=100,  
        learning_rate=2e-4,
        save_on_each_node=True,  
        gradient_checkpointing=True,  
        report_to="none" 
    )


def predict_answer(model, tokenizer, context, question):
    # 构建对话消息列表
    messages = [
        {"role": "system", "content": "现在进行知识问答，要求根据提供的段落回答问题，要求只给出答案即可。"},  # 系统提示
        {"role": "user", "content": f"基于这一段文本: {context}, 回答问题：{question}"}   # 用户输入
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  
    )


    model_inputs = tokenizer([formatted_text], return_tensors="pt").to("mps")

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=50,                  
            do_sample=False,                    
            temperature=0.0,                    
            top_p=0.95,                         
            repetition_penalty=1.5,             
            pad_token_id=tokenizer.pad_token_id,  
            eos_token_id=tokenizer.eos_token_id  
        )


    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()  


# 批量预测
def batch_predict(model, tokenizer, contexts, questions):
    pred_answers = []

    # 遍历所有测试文本，显示进度条
    for context, question in tqdm(contexts, questions, desc="批量知识问答"):
        try:
            pred_answer = predict_answer(model, tokenizer, context, question)
            pred_answers.append(pred_answer)
        except Exception as e:
            print(f"预测问题 '{question}' 时出错: {e}")
            pred_answers.append("")  # 出错时添加空字符串

    return pred_answers


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    ds_train, ds_dev, dev_paragraphs, dev_questions, dev_answers = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "/Users/wangyingyue/materials/大模型学习资料——八斗/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    train_tokenized = ds_train.map(process_func_with_tokenizer, remove_columns=ds_train.column_names)
    dev_tokenized = ds_dev.map(process_func_with_tokenizer, remove_columns=ds_dev.column_names)

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
        eval_dataset=dev_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    # trainer.save_model()
    model.save_pretrained("./qa-qwen_lora-model/")  # 保存模型
    tokenizer.save_pretrained("./qa-qwen_lora-model/")  # 保存分词器



# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "/Users/wangyingyue/materials/大模型学习资料——八斗/models/Qwen/Qwen3-0.6B"
    tokenizer, base_model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重 - 使用正确的方式
    adapter_path = "./qa-qwen_lora-model/"

    # 检查适配器文件是否存在
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"适配器配置文件不存在于: {adapter_path}")
        return

    # 使用 PeftModel.from_pretrained 加载适配器
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to("mps")
        trust_remote_code = True
        print("成功加载LoRA适配器")
    except Exception as e:
        print(f"加载适配器失败: {e}")
        return

    ds_train, ds_dev, dev_paragraphs, dev_questions, dev_answers = load_and_preprocess_data()
    # 在验证集上测试几个样本
    print("\n在验证集上测试:")
    for i in range(min(3, len(dev_paragraphs))):
        context = dev_paragraphs[i]
        question = dev_questions[i]
        expected_answer = dev_answers[i]

        predicted_answer = predict_answer(model, tokenizer, context, question)

        print(f"问题 {i + 1}: {question}")
        print(f"预期答案: {expected_answer}")
        print(f"预测答案: {predicted_answer}")
        print(f"匹配: {expected_answer == predicted_answer}")
        print()


if __name__ == "__main__":
    # main()
    test_single_example()

