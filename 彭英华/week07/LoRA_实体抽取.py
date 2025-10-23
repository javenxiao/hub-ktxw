import codecs
from tqdm import tqdm
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq


def load_and_preprocess_data():
    """加载和预处理数据"""
    train_lines = codecs.open('./msra/train/sentences.txt', encoding='utf-8').readlines()
    train_lines = [x.replace(' ', '').strip() for x in train_lines]
    # train_data = pd.read_csv('intent-dataset.csv', sep='\t', header=None)
    train_tags = codecs.open('./msra/train/tags.txt').readlines()
    train_tags = [x.strip().split(' ') for x in train_tags]
    train_entities = []
    for tokens, labels in zip(train_lines, train_tags):
        entities = []
        current_entity = ""
        current_type = ""
        tokens = list(tokens)
        for token,label in zip(tokens,labels):
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = token
                current_type = label[2:]
            elif label.startswith('I-') and current_entity and current_type == label[2:]:
                current_entity += token
            else:
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""
                if label.startswith('B-'):
                    current_entity = token
                    current_type = label[2:]

        if current_entity:
           entities.append((current_entity, current_type))
        train_entities.append(entities)
    data=[]
    for text,label in zip(train_lines,train_entities):
        data.append({'instruction':text,
                           'output':label
        })
    train_data = pd.DataFrame(data)
    # 重命名列并添加输入列
    train_data["input"] = ""
    # train_data.columns = ["instruction", "output", "input"]

    # 转换为Hugging Face Dataset
    ds = Dataset.from_pandas(train_data)

    return ds
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
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    instruction_text = f"<|im_start|>system\n现在进行实体抽取任务<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    if len(example['output'])==0:
        example['output'] = "文本中没有识别到命名实体"
    else:
        entity_strings = []
        for entity in example['output']:
            if len(entity) >= 2:
                entity_text = str(entity[0])
                entity_type = str(entity[1])
                entity_strings.append(f"{entity_text}({entity_type})")

        example['output'] = "、".join(entity_strings)
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
def predict_intent(model, tokenizer, text, device='cpu'):
    """预测单个文本的意图"""
    messages = [
        {"role": "system", "content": "现在进行实体识别任务"},
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
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图"""
    pred_labels = []

    for text in tqdm(test_texts, desc="实体识别"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"实体识别 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels
def porcess (example):
    if len(example['output'])==0:
        example['output'] = "文本中没有识别到命名实体"
    else:
        entity_strings = []
        for entity in example['output']:
            if len(entity) >= 2:
                entity_text = str(entity[0])
                entity_type = str(entity[1])
                entity_strings.append(f"{entity_text}({entity_type})")

        example['output'] = "、".join(entity_strings)
    return example
print("加载数据...")
ds = load_and_preprocess_data()

# 2. 初始化模型和tokenizer
print("初始化模型和tokenizer...")
model_path = "./models/Qwen/Qwen3-0.6B"
tokenizer, model = initialize_model_and_tokenizer(model_path)
#
# # 3. 处理数据
print("处理训练数据...")
process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)
#
# 4. 划分训练集和验证集
train_ds = Dataset.from_pandas(ds.to_pandas().iloc[:200])
eval_ds = Dataset.from_pandas(ds.to_pandas()[-200:])
#
train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)
#
# 5. 设置LoRA
print("设置LoRA...")
model.enable_input_require_grads()
model = setup_lora(model)
#
# 6. 配置训练参数
print("配置训练参数...")
training_args = setup_training_args()
#
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
model.load_adapter("./output_Qwen1.5/checkpoint-45")
# model.cpu()
test_sentences = [
    '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
    '人工智能是未来的希望，也是中国和美国的冲突点。',
    '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
    '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
    '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
    '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
]
print(batch_predict(model,tokenizer,test_sentences,device='cuda'))
