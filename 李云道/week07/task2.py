import codecs
from typing import Literal

from datasets import Dataset  # Hugging Face数据集类，提供高效的数据加载和处理功能
from transformers import (
    AutoTokenizer,  # 自动选择适合预训练模型的分词器
    AutoModelForCausalLM,  # 自动选择适合的自回归语言模型
    DataCollatorForSeq2Seq,  # 用于序列到序列任务的数据整理器
    TrainingArguments,  # 训练参数配置类
    Trainer,  # 训练器类，封装训练循环
)

# pip install peft
from peft import LoraConfig, TaskType, get_peft_model  # 参数高效微调库(LoRA)
from tqdm import tqdm  # 进度条显示库
import torch  # PyTorch深度学习框架

'''
用Qwen-LoRA方法，微调一个识别模型，数据集参考：04_BERT实体抽取.py
'''

# 定义标签类型
# tag_type = codecs.open("asset/msra/tags.txt", encoding='utf-8').read().strip().split('\n')
# id2label = {i: label for i, label in enumerate(tag_type)}
# label2id = {label: i for i, label in enumerate(tag_type)}


# 数据加载和预处理
def load_and_preprocess_data(typing: Literal["text", "train", "val"]) -> Dataset:
    """加载和预处理数据
    """
    lines = codecs.open(f'asset/msra/{typing}/sentences.txt', encoding='utf-8').readlines()
    lines = [line.replace(" ", "").strip() for line in lines]

    tags = codecs.open(f'asset/msra/{typing}/tags.txt', encoding='utf-8').readlines()
    # tags = [[label2id[i] for i in tag.strip().split(" ")] for tag in tags]

    # 序列标注文本格式
    output = []
    for i, tag in enumerate(tags):
        discover = []
        for j, t in enumerate(tag.strip().split(" ")):
            discover.append(lines[i][j] + "/" + tags[i][j] + " ")
        output.append("".join(discover))

    """ 以下式适用于判别式模型（如BERT+Linear），但不适合生成式模型
    {'input': '如何解决足球界长期存在的诸多矛盾，重振昔日津门足球的雄风，成为天津足坛上下内外到处议论的话题。',
     'output': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 6, 4,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    """

    # 转换为Hugging Face Dataset格式
    ds = Dataset.from_dict({
        "input": lines, "output": output
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
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式，并添加适当的特殊标记

    参数:
        example: 单个数据样本，包含instruction, output和input字段
        tokenizer: 分词器实例
        max_length: 最大序列长度，超过将被截断

    返回:
        包含input_ids, attention_mask和labels的字典
    """
    # 构建指令部分（使用Qwen模型的特殊标记格式）
    instruction_text = f"<|im_start|>system\n请对以下文本进行命名实体识别，使用BIO标注格式[O,B-ORG,I-PER,B-PER,I-LOC,I-ORG,B-LOC]<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)  # 不添加特殊标记（已手动添加）

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]  # 添加填充标记
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 添加注意力掩码

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


def compute_metrics(p):
    predictions, labels = p
    # TODO
    pass


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
        # do_eval=True,  # 是否进行评估
        eval_steps=50,  # 评估步数间隔
        num_train_epochs=5,  # 训练轮数
        save_steps=50,  # 模型保存步数间隔
        learning_rate=1e-4,  # 学习率
        save_on_each_node=True,  # 在每个节点上保存模型
        gradient_checkpointing=True,  # 使用梯度检查点节省内存
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_intent(model, tokenizer, text, device='cuda'):
    """预测单个文本的意图
    """
    # 构建对话消息格式
    messages = [
        {"role": "system", "content": "请对以下文本进行命名实体识别，使用BIO标注格式[O,B-ORG,I-PER,B-PER,I-LOC,I-ORG,B-LOC]"},
        {"role": "user", "content": text}
    ]

    # 应用聊天模板格式化消息
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 不进行分词
        add_generation_prompt=True  # 添加生成提示
    )

    # Tokenize输入文本
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,  # 最大生成token数
            do_sample=True,  # 使用采样而非贪婪解码
            temperature=0.1,  # 采样温度（较低值使输出更确定）
            top_p=0.9,  # 使用top-p采样
            repetition_penalty=1.2,  # 添加重复惩罚
            pad_token_id=tokenizer.pad_token_id,  # 填充token ID
            eos_token_id=tokenizer.eos_token_id  # 结束token ID
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()  # 返回清理后的响应


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
    ds = load_and_preprocess_data("train")
    # val_ds = load_and_preprocess_data("val")
    print(ds[0])

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
    train_ds = Dataset.from_pandas(ds.to_pandas().iloc[:200])  # 前200个样本作为训练集
    val_ds = Dataset.from_pandas(ds.to_pandas()[-50:])  # 后200个样本作为验证集

    # 处理训练集和验证集
    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = val_ds.map(process_func_with_tokenizer, remove_columns=val_ds.column_names)
    print(train_tokenized[0])

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
    test_text = """
    如 何 解 决 足 球 界 长 期 存 在 的 诸 多 矛 盾 ， 重 振 昔 日 津 门 足 球 的 雄 风 ， \
    成 为 天 津 足 坛 上 下 内 外 到 处 议 论 的 话 题 。
    """.strip().replace(" ","")

    result = predict_intent(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"预测意图: {result}")


if __name__ == "__main__":
    model_path = r"D:\Download\Qwen\Qwen3-0.6B"  # Qwen 0.6B模型路径
    lora_path = "asset/lora_task2"

    # 执行主函数训练
    main(model_path, lora_path)

    # 单独测试
    do_single_example(model_path, lora_path)
