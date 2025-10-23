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
import codecs
import warnings

warnings.filterwarnings('ignore')
import os
print(f"当前工作目录: {os.getcwd()}")
# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 数据加载和预处理
def tags_to_entities(text, tags):
    """将标签序列转换为实体描述文本"""
    entities = []
    current_entity = ""
    current_type = ""

    for char, tag in zip(text, tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append(f"{current_entity}:{current_type}")
            current_entity = char
            current_type = tag[2:]
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
            current_entity += char
        else:
            if current_entity:
                entities.append(f"{current_entity}:{current_type}")
            current_entity = ""
            current_type = ""

    # 处理句子末尾的实体
    if current_entity:
        entities.append(f"{current_entity}:{current_type}")

    # 格式化实体列表
    if entities:
        return ", ".join(entities)
    return "无实体"


def load_and_preprocess_data():
    """加载和预处理MSRA实体抽取数据集"""
    # 加载训练数据
    train_lines = codecs.open(
        './msra/train/sentences.txt').readlines()[:1000]  # 限制数据量以加快训练
    train_lines = [x.replace(' ', '').strip() for x in train_lines]

    train_tags = codecs.open('./msra/train/tags.txt').readlines()[:1000]
    train_tags = [x.strip().split(' ') for x in train_tags]

    # 将标签转换为实体描述文本
    train_entities = [
        tags_to_entities(text, tags)
        for text, tags in zip(train_lines, train_tags)
    ]

    # 加载验证数据
    val_lines = codecs.open('./msra/val/sentences.txt').readlines()[:100]
    val_lines = [x.replace(' ', '').strip() for x in val_lines]

    val_tags = codecs.open('./msra/val/tags.txt').readlines()[:100]
    val_tags = [x.strip().split(' ') for x in val_tags]

    # 将验证集标签转换为实体描述文本
    val_entities = [
        tags_to_entities(text, tags)
        for text, tags in zip(val_lines, val_tags)
    ]

    # 创建训练和验证数据集
    train_ds = Dataset.from_dict({
        "instruction": train_lines,
        "output": train_entities
    })

    eval_ds = Dataset.from_dict({
        "instruction": val_lines,
        "output": val_entities
    })

    return train_ds, eval_ds

# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型，并处理MPS设备兼容性"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 处理MPS设备的特殊情况
    if device.type == 'mps':
        # MPS设备上先加载到CPU，然后转为float32以避免BFloat16错误
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # 先加载到CPU
            torch_dtype=torch.float32,  # 使用float32避免MPS BFloat16错误
            trust_remote_code=True
        )
        # 手动移到MPS设备
        model.to(device)
    else:
        # 其他设备可以使用auto映射和float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    return tokenizer, model

# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    """将指令和输出转换为模型训练格式"""
    # 实体抽取系统提示
    system_prompt = "你是一个实体抽取助手，请从用户提供的文本中提取实体，并按照'实体:类型'的格式输出，多个实体之间用逗号分隔。"

    # 构建ChatML格式的输入
    instruction_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [
        tokenizer.pad_token_id
    ]
    attention_mask = instruction["attention_mask"] + response[
        "attention_mask"] + [1]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [
        tokenizer.pad_token_id
    ]

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

# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_Qwen3_ner",
        per_device_train_batch_size=4,  # 减小批次大小以适应内存
        gradient_accumulation_steps=4,
        logging_steps=50,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=3,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"
    )

# 实体抽取预测函数
def predict_entities(model, tokenizer, text, device='cpu'):
    """预测文本中的实体"""
    messages = [{
        "role":
        "system",
        "content":
        "你是一个实体抽取助手，请从用户提供的文本中提取实体，并按照'实体:类型'的格式输出，多个实体之间用逗号分隔。"
    }, {
        "role": "user",
        "content": text
    }]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(messages,
                                                   tokenize=False,
                                                   add_generation_prompt=True)

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    # 生成预测 - 改进生成策略
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=128,  # 减少最大生成长度，因为实体抽取结果通常不会太长
            do_sample=False,  # 关闭随机采样，使用贪婪解码
            temperature=0.0,  # 确定性生成
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            repetition_penalty=1.1  # 避免重复生成
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids,
                                      skip_special_tokens=True)[0]
    # 后处理：清理输出文本，移除明显的无意义内容
    if response:
        # 移除Qwen模型可能产生的特殊标记
        response = response.replace("Question", "").replace("Instructions", "").replace("Answer", "").replace("*", "").strip()
        # 如果结果包含中文字符，尝试提取有效的实体部分
        import re
        # 匹配类似"实体:类型"的模式
        entities = re.findall(r'[\u4e00-\u9fa5]+:[A-Z]+', response)
        if entities:
            return ", ".join(entities)
    return response.strip()

# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_ds, eval_ds = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

    # 4. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 5. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 6. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    trainer.train()

    # 7. 保存模型
    print("保存模型...")
    trainer.save_model("./output_Qwen3_ner")
    tokenizer.save_pretrained("./output_Qwen3_ner")

    # 8. 测试预测
    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。'
    ]

    print("\n测试实体抽取结果:")
    for sentence in test_sentences:
        result = predict_entities(model, tokenizer, sentence, device)
        print(f"句子: {sentence}")
        print(f"实体: {result}")
        print()

# 单独测试函数
def test_single_example():
    model_path = "../models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    try:
        # 尝试加载最新的checkpoint
        import os
        import glob
        checkpoint_dirs = glob.glob("./output_Qwen3_ner/checkpoint-*")
        if checkpoint_dirs:
            # 选择最后一个checkpoint
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
            print(f"加载checkpoint: {latest_checkpoint}")
            model.load_adapter(latest_checkpoint)
        else:
            # 尝试加载主目录
            model.load_adapter("./output_Qwen3_ner/")
    except Exception as e:
        print(f"加载适配器失败: {e}")
        print("请先运行main()函数进行训练")
        return

    model.to(device)

    # 测试预测
    test_text = "今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。"
    result = predict_entities(model, tokenizer, test_text, device)
    print(f"输入: {test_text}")
    print(f"实体: {result}")

if __name__ == "__main__":
    # 执行主函数进行训练
    main()

    # 或者单独测试
    # test_single_example()
