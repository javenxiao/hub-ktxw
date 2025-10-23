import json
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from tqdm import tqdm

def load_and_preprocess_data(path):
    formatted_data = []
    train = json.load(open(path,encoding='utf-8'))
    for article in train["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer_info = qa["answers"][0]  # 获取完整的答案信息
                answer_text = answer_info["text"]
                answer_start = answer_info["answer_start"]
                formatted_data.append({
                    "instruction": f"文本：{context}\n问题：{question}",
                    "output": f"答案：{answer_text}（位于文本第{answer_start}到{answer_start+len(answer_text)}字符处）",
                })
    train_data = pd.DataFrame(formatted_data)
    train_data["input"] = ""
    ds = Dataset.from_pandas(train_data)

    return ds
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,   #不使用快速版本的
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16  # 使用半精度减少内存占用
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    instruction_text = f"<|im_start|>system\n现在进行知识问答任务，根据后面的文本对问题进行回答<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
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
        output_dir="./output_Qwen1.5_qa",
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


# 预测函数
def predict_intent(model, tokenizer, text, device='cpu'):
    """预测单个文本的意图"""
    messages = [
        {"role": "system", "content": "现在进行知识问答任务，根据后面的文本对问题进行回答"},
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


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图"""
    pred_labels = []

    for text in tqdm(test_texts, desc="知识问答"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels
print("加载数据...")
ds = load_and_preprocess_data('./cmrc2018_public/train.json')

# 2. 初始化模型和tokenizer
print("初始化模型和tokenizer...")
model_path = "./models/Qwen/Qwen3-0.6B"
tokenizer, model = initialize_model_and_tokenizer(model_path)
# 3. 处理数据
print("处理训练数据...")
process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

# 4. 划分训练集和验证集
train_ds = Dataset.from_pandas(ds.to_pandas().iloc[:200])
eval_ds = Dataset.from_pandas(ds.to_pandas()[-200:])

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
model.load_adapter('./output_Qwen1.5_qa/checkpoint-45')

test_text = f"文本：罗亚尔港号（USS Port Royal CG-73）是美国海军提康德罗加级导弹巡洋舰，是该级巡洋舰的第27艘也是最后一艘。它也是美国海军第二艘以皇家港为名字命名的军舰。第一艘是1862年下水、曾参与南北战争的。船名来自曾在美国独立战争和南北战争中均发生过海战的南卡罗来纳州（Port Royal Sound）。美国海军在1988年2月25日订购该船，1991年10月18日在密西西比州帕斯卡古拉河畔的英戈尔斯造船厂放置龙骨。1992年11月20日下水，1992年12月5日由苏珊·贝克（Susan G. Baker，老布什政府时期的白宫办公厅主任，也是前国务卿詹姆斯·贝克的夫人）为其命名，1994年7月9日正式服役。2009年2月5日，罗亚尔港号巡洋舰在位于檀香山国际机场以南0.5英里的一处珊瑚礁上发生搁浅，之前该舰刚完成在旱坞内的维护，正在进行维护后的第一次海试。2009年2月9日凌晨2点，罗亚尔港号被脱离珊瑚礁。无人在这次事故中受伤，也未发生船上燃料的泄漏。但由于这次搁浅，罗亚尔港号巡洋舰不得不回到旱坞重新进行维修。1995年12月加入尼米兹号为核心的航空母舰战斗群，参与了南方守望行动，这是罗亚尔港号巡洋舰首次参与的军事部署行动。1996年3月由于台湾海峡导弹危机的发生被部署到了南中国海，随着危机的结束，1997年9月至1998年3月回到尼米兹号航空母舰战斗群参与南方守望行动。后随约翰·C·斯坦尼斯号航空母舰战斗群继续参加南方守望行动。2000年1月由于多次追击涉嫌违反联合国禁运制裁走私偷运伊拉克原油的船只因而造成对船上动力设备的持续性机械磨损而撤离，回到夏威夷进行整修和升级。2001年11月7日加入约翰·C·斯坦尼斯号航空母舰战斗群参与旨在对基地组织和对它进行庇护的阿富汗塔利班政权进行打击的持久自由军事行动。\n问题：罗亚尔港号是哪一年下水的？"
print(predict_intent(model,tokenizer,test_text,device='cuda'))
