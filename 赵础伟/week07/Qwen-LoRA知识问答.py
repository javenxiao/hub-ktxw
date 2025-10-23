import codecs
import json
import re
from typing import Literal
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm  # 进度条显示库
import torch  # PyTorch深度学习框架
from datasets import Dataset  # Hugging Face数据集类，提供高效的数据加载和处理功能
from transformers import (
    AutoTokenizer,  # 自动选择适合预训练模型的分词器
    AutoModelForCausalLM,  # 自动选择适合的自回归语言模型
    DataCollatorForSeq2Seq,  # 用于序列到序列任务的数据整理器
    TrainingArguments,  # 训练参数配置类
    Trainer,  # 训练器类，封装训练循环
)

# 数据加载和预处理
def load_and_preprocess_data(typing: Literal["dev", "train"]) -> Dataset:
    """加载和预处理数据
    """
    data = json.load(codecs.open(f'cmrc2018_public/{typing}.json', encoding='utf-8'))

    paragraphs, questions, answers = [], [], []
    for content in data["data"]:
        context = content["paragraphs"][0]["context"]
        for qas in content["paragraphs"][0]["qas"]:
            # 获取数据
            question = qas["question"]
            answer_start = qas["answers"][0]["answer_start"]
            text = qas["answers"][0]["text"]

            paragraphs.append(context)
            questions.append(question)
            answers.append({
                "answer_start": answer_start,
                "text": text
            })

    ds = Dataset.from_dict({
        "paragraph": paragraphs,
        "question": questions,
        "answer": answers
    })

    return ds


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型
    加载预训练模型和对应的分词器
    参数:
        model_path: 预训练模型的本地路径或Hugging Face模型标识
    返回:
        tokenizer: 分词器实例
        model: 模型实例
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,  # 不使用快速分词器
        trust_remote_code=True  # 信任远程代码（对于某些自定义模型需要）
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # 自动分配设备（CPU/GPU）
        dtype=torch.float16  # 使用半精度浮点数减少内存占用
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=1024):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式，并添加适当的特殊标记
    """
    # 构建对话消息
    messages = [
        {"role": "system", "content": "你是一个中文问答助手，请根据提供的文本内容，用简洁准确的中文回答问题。"},
        {"role": "user", "content": f"文本内容：{example['paragraph'][:800]}\n问题：{example['question']}"},
        {"role": "assistant", "content": example['answer']['text']}
    ]

    # 使用apply_chat_template自动处理格式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # 注意这里改为False，因为我们已经提供了完整对话
    )

    # Tokenize整个文本
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )

    # 创建标签 - 更简单的方法
    # 复制input_ids作为labels的基础
    labels = tokenized["input_ids"].copy()

    # 找到assistant开始的位置
    assistant_start = text.find("<|im_start|>assistant")
    if assistant_start != -1:
        # Tokenize到assistant之前的部分
        prefix = text[:assistant_start]
        prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]

        labels[:len(prefix_tokens)] = [-100] * len(prefix_tokens)

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型
    使用LoRA进行参数高效微调，只训练少量参数
    参数:
        model: 需要应用LoRA的模型
    返回:
        应用了LoRA的模型
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型为因果语言模型
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 要应用LoRA的模块
        inference_mode=False,  # 训练模式
        r=8,  # LoRA秩（rank）
        lora_alpha=32,  # LoRA alpha参数
        lora_dropout=0.1  # LoRA dropout率
    )

    # 将LoRA适配器应用到模型
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # 打印可训练参数信息

    return model

# 训练配置
def setup_training_args(lora_path):
    """设置训练参数
    配置训练过程中的各种超参数和策略
    返回:
        TrainingArguments实例
    """
    return TrainingArguments(
        output_dir=lora_path,  # 输出目录
        per_device_train_batch_size=6,  # 每个设备的训练批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数（模拟更大批次）
        logging_steps=100,  # 日志记录步数间隔
        do_eval=True,  # 是否进行评估
        eval_steps=50,  # 评估步数间隔
        num_train_epochs=5,  # 训练轮数
        save_steps=50,  # 模型保存步数间隔
        learning_rate=1e-4,  # 学习率
        save_on_each_node=True,  # 在每个节点上保存模型
        gradient_checkpointing=True,  # 使用梯度检查点节省内存
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_intent(model, tokenizer, paragraph, question, device='cuda'):
    """预测单个文本的意图
    """
    # 使用与训练时完全相同的提示格式
    messages = [
        {"role": "system", "content": "你是一个中文问答助手，请根据提供的文本内容，用简洁准确的中文回答问题。"},
        {"role": "user", "content": f"文本内容：{paragraph[:500]}\n问题：{question}"}
    ]

    # 应用相同的模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize输入文本
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测 - 使用更严格的参数
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.3,  # 降低温度，减少随机性
            top_p=0.85,
            top_k=50,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 清理响应 - 移除可能的标记
    response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    response = re.sub(r"<think>.*</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"\s+", "", response, flags=re.DOTALL)

    return response


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图
    对多个文本进行意图分类预测
    参数:
        model: 训练好的模型
        tokenizer: 分词器
        test_texts: 待预测的文本列表
        device: 设备类型
    返回:
        预测的意图标签列表
    """
    pred_labels = []

    # 使用进度条遍历所有测试文本
    for text in tqdm(test_texts, desc="预测意图"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main(model_path, lora_path):
    """主执行函数
    完整的训练流程：数据加载、模型初始化、训练、评估
    """
    # 1. 加载数据
    print("加载数据...")
    train_ds = load_and_preprocess_data("train")
    # val_ds = load_and_preprocess_data("dev")
    # print(len(train_ds), train_ds[0])

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    # 创建带有固定tokenizer的处理函数
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    # 应用处理函数到整个数据集
    # tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

    # 4. 划分训练集和验证集 跑的比较慢，少跑点儿吧
    train_ds = Dataset.from_pandas(train_ds.to_pandas().iloc[:400])  # 前200个样本作为训练集
    val_ds = Dataset.from_pandas(train_ds.to_pandas()[-200:])  # 后200个样本作为验证集

    # 处理训练集和验证集
    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = val_ds.map(process_func_with_tokenizer, remove_columns=val_ds.column_names)
    # print(train_tokenized[0])

    # 5. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()  # 启用输入梯度要求
    model = setup_lora(model)  # 应用LoRA配置

    # 6. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args(lora_path)

    # 7. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,  # 训练数据集
        eval_dataset=eval_tokenized,  # 验证数据集
        data_collator=DataCollatorForSeq2Seq(  # 数据整理器
            tokenizer=tokenizer,
            padding=True,  # 启用填充
            pad_to_multiple_of=8  # 填充到8的倍数，优化GPU内存使用
        ),
    )

    trainer.train()  # 开始训练

    # 8. 保存模型（注释掉了，实际使用时取消注释）
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(lora_path)

    # 评估模型性能
    print("评估模型...")
    eval_results = trainer.evaluate()
    print(f"评估结果: {eval_results}")


# 单独测试函数
def do_single_example(model_path, lora_path):
    """单独测试函数
    加载训练好的模型并对单个示例进行预测
    """
    # 下载模型的命令（注释形式）
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B

    # 初始化模型和分词器
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter(lora_path)
    model = model.cuda()

    # 测试预测
    test_ds = load_and_preprocess_data("dev")
    # test_ds.shuffle()
    test_ds = Dataset.from_pandas(test_ds.to_pandas().iloc[:10])  # 前3个样本作为训练集

    for i in range(min(3, len(test_ds))):
        paragraph = test_ds["paragraph"]
        question = test_ds[i]["question"]
        expected_answer = test_ds[i]["answer"]['text']

        predicted_answer = predict_intent(model, tokenizer, paragraph, question)

        print(f"问题 {i + 1}: {question}")
        print(f"预期答案: {expected_answer}")
        print(f"预测答案: {predicted_answer}")
        print(f"匹配: {expected_answer == predicted_answer}")  # 检查是否匹配
        print()


if __name__ == "__main__":
    model_path = r"C:\Users\16406\.ollama\models\Qwen\Qwen3-4b"  # Qwen 0.6B模型路径
    lora_path = "blobs"

    # 执行主函数训练
    main(model_path, lora_path)

    # 单独测试
    do_single_example(model_path, lora_path)