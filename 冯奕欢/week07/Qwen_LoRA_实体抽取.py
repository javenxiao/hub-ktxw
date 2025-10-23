import codecs
import warnings

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 加载训练数据
train_lines = codecs.open('msra/train/sentences.txt', encoding='utf-8').readlines()[:1000]
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags = codecs.open('msra/train/tags.txt', encoding='utf-8').readlines()[:1000]
train_tags = [x.strip().split(' ') for x in train_tags]

# 加载验证数据
val_lines = codecs.open('msra/val/sentences.txt', encoding='utf-8').readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('msra/val/tags.txt', encoding='utf-8').readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]


# 准备数据
def prepare_dataset(texts, tags):
    tokens = [list(text) for text in texts]
    labels = list()
    for token, tag in zip(tokens, tags):
        temp = list()
        for key, value in zip(token, tag):
            temp.append(f"{key}:{value}")
        temp = " ".join(temp)
        labels.append(temp)
    dataset = Dataset.from_dict({
        "texts": texts,
        "labels": labels
    })
    return dataset


# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n现在进行实体抽取任务，需为文本中的每个字符标注 BIO 标签。实体类型包括：ORG（组织）、PER（人名）、LOC（地点）。输出格式：字符1:标签1 字符2:标签2 ... 字符N:标签N<|im_end|>\n<|im_start|>user\n{example['texts']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    print('instruction -> ', instruction_text)

    # 构建响应部分
    response_text = f"{example['labels']}"
    response = tokenizer(response_text, add_special_tokens=False)

    print('response -> ', response_text)

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
        output_dir="./output_Qwen1.5",
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


# 主函数
def main():

    """主执行函数"""

    # 1. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 2. 准备训练和验证数据集
    train_dataset = prepare_dataset(train_lines, train_tags)
    eval_dataset = prepare_dataset(val_lines, val_tags)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    train_tokenized = train_dataset.map(process_func_with_tokenizer, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(process_func_with_tokenizer, remove_columns=eval_dataset.column_names)

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
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    # 7. 训练
    trainer.train()

    # 8. 保存模型
    # print("保存模型...")
    # trainer.save_model()
    # tokenizer.save_pretrained("./output_Qwen")


# 预测函数
def predict_ner(model, tokenizer, text, device='cpu'):
    """预测实体"""
    messages = [
        {"role": "system", "content": "现在进行实体抽取任务，需为文本中的每个字符标注 BIO 标签。实体类型包括：ORG（组织）、PER（人名）、LOC（地点）。输出格式：字符1:标签1 字符2:标签2 ... 字符N:标签N"},
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


# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "../models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_Qwen1.5/")
    model.cpu()

    # 测试预测
    test_text = "北京的许多眼科专家走出医院，一边义诊、一边向群众宣讲预防眼外伤。"
    result = predict_ner(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"实体标注: {result}")


if __name__ == "__main__":
    # 执行主函数
    main()
    # 预测
    test_single_example()