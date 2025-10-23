import codecs
import json
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

# 加载数据
train = json.load(open('cmrc2018_public/train.json', encoding='utf-8'))
dev = json.load(open('cmrc2018_public/dev.json', encoding='utf-8'))


# 准备数据
def prepare_dataset(data):
    paragraphs = []
    questions = []
    answers = []
    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            paragraphs.append(context)
            questions.append(qa['question'])
            answers.append(qa['answers'][0]['text'])
    dataset = Dataset.from_dict({
        "paragraphs": paragraphs[:100],
        "questions": questions[:100],
        "answers": answers[:100]
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
    instruction_text = f"""<|im_start|>system
任务：从用户提供的上下文文本中，抽取能回答问题的连续文本作为答案。
要求：仅输出答案文本，不要包含任何额外内容（如格式符号、解释）。
<|im_end|>
<|im_start|>user
上下文：{example['paragraphs']}
问题：{example['questions']}
<|im_end|>
<|im_start|>assistant
"""
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    print("instruction ->", instruction_text)

    # 构建响应部分
    response_text = f"{example['answers']}"
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
    train_dataset = prepare_dataset(train)
    eval_dataset = prepare_dataset(dev)

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
def predict_answer(model, tokenizer, paragraph, question, device='cpu'):
    """预测答案"""
    messages = [
        {"role": "system", "content": """任务：从用户提供的上下文文本中，抽取能回答问题的连续文本作为答案。
要求：仅输出答案文本，不要包含任何额外内容（如格式符号、解释）。"""},
        {"role": "user", "content": f"""上下文：{paragraph}
问题：{question}"""}
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
    test_paragraph = "为日本漫画足球小将翼的一个角色，自小父母离异，与父亲一起四处为家，每个地方也是待一会便离开，但他仍然能够保持优秀的学业成绩。在第一次南葛市生活时，与同样就读于南葛小学的大空翼为黄金拍档，曾效力球队包括南葛小学、南葛高中、日本少年队、日本青年军、日本奥运队。效力日本青年军期间，因救同母异父的妹妹导致被车撞至断脚，在决赛周只在决赛的下半场十五分钟开始上场，成为日本队夺得世青冠军的其中一名功臣。基本资料绰号：球场上的艺术家出身地：日本南葛市诞生日：5月5日星座：金牛座球衣号码：11担任位置：中场、攻击中场、右中场擅长脚：右脚所属队伍：盘田山叶故事发展岬太郎在小学期间不断转换学校，在南葛小学就读时在全国大赛中夺得冠军；国中三年随父亲孤单地在法国留学；回国后三年的高中生涯一直输给日本王牌射手日向小次郎率领的东邦学院。在【Golden 23】年代，大空翼、日向小次郎等名将均转战海外，他与松山光、三杉淳组成了「3M」组合（松山光Hikaru Matsuyama、岬太郎Taro Misaki、三杉淳Jyun Misugi）。必杀技1. 回力刀射门2. S. S. S. 射门3. 双人射门(与大空翼合作)"
    test_question = "岬太郎在第一次南葛市生活时的搭档是谁"
    result = predict_answer(model, tokenizer, test_paragraph, test_question)
    print(f"上下文: {test_paragraph}")
    print(f"问题: {test_question}")
    print(f"回答: {result}")


if __name__ == "__main__":
    # 执行主函数
    main()
    # 预测
    test_single_example()