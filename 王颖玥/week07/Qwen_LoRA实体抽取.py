import codecs  
import numpy as np
import pandas as pd

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

warnings.filterwarnings('ignore') 


# 定义标签类型（NER任务的核心：给每个字打标签）
# O：不是实体；B-ORG：机构名开头；I-ORG：机构名中间；B-PER：人名开头；
# I-PER：人名中间；B-LOC：地名开头；I-LOC：地名中间
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

def load_and_preprocess_data():
    # 加载训练数据
    train_lines = codecs.open('./msra/train/sentences.txt').readlines()[:1000]

    train_lines = [x.replace(' ', '').strip() for x in train_lines]

    train_tags = codecs.open('./msra/train/tags.txt').readlines()[:1000]
    train_tags = [x.strip().split(' ') for x in train_tags]  # 处理标签：按空格分割成列表

    # 加载验证数据
    val_lines = codecs.open('./msra/val/sentences.txt').readlines()[:100]
    val_lines = [x.replace(' ', '').strip() for x in val_lines]

    val_tags = codecs.open('./msra/val/tags.txt').readlines()[:100]
    val_tags = [x.strip().split(' ') for x in val_tags]

    train_data = []
    # 直接先从文本中抽取实体
    for text, labels in zip(train_lines, train_tags):
        entities = extract_entities(text, labels)
        if entities:
            output = "\n".join([f"{entity_type}: {entity}" for entity, entity_type in entities])
        else:
            output = "未识别到实体"
        train_data.append(
            {
                "instruction": f"现在进行实体抽取任务，类型只能是ORG（机构），PER（人名），LOC（地名），直接输出结果，格式为'类型：实体'，识别不出实体就输出'未识别到实体': {text}",
                "output": output,
                "input": ""
            }
        )

    val_data = []
    for text, labels in zip(val_lines, val_tags):
        entities = extract_entities(text, labels)
        val_data.append(
            {
                "instruction": f"现在进行实体抽取任务，类型只能是ORG（机构），PER（人名），LOC（地名），直接输出结果，格式为'类型：实体'，识别不出实体就输出'未识别到实体': {text}",
                "output": "\n".join([f"{entity_type}: {entity}" for entity, entity_type in entities]) if entities else "未识别到实体",
                "input": ""
            }
        )

    ds_train = Dataset.from_pandas(pd.DataFrame(train_data))
    ds_val = Dataset.from_pandas(pd.DataFrame(val_data))

    return ds_train, ds_val

# 提取实体
def extract_entities(text, labels):
    entities = []  
    current_entity = ""
    current_type = ""

    for token, label in zip(text, labels):
        if label.startswith('B-'):  # 遇到实体开头（比如B-PER、B-LOC）
            if current_entity:  # 如果之前有未完成的实体，先存起来
                entities.append((current_entity, current_type))
            current_entity = token  # 开始新实体
            current_type = label[2:]  # 取类型（比如B-PER→PER）
        elif label.startswith('I-') and current_entity and current_type == label[2:]:  # 实体中间（比如I-PER，且和当前类型一致）
            current_entity += token  # 拼接字符
        else:  # 非实体（O）或类型不一致
            if current_entity:  # 如果有未完成的实体，存起来
                entities.append((current_entity, current_type))
            current_entity = ""
            current_type = ""
            # 特殊情况：如果当前标签是B-开头（比如前面漏了）
            if label.startswith('B-'):
                current_entity = token
                current_type = label[2:]

    # 最后检查是否有剩余实体
    if current_entity:
        entities.append((current_entity, current_type))

    return entities

# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,  # 不使用快速分词器
        trust_remote_code=True  # 信任远程代码（因为有些模型有自定义代码）
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=None,  # 不自动分配设备
        torch_dtype=torch.float16,  # 使用半精度减少内存占用
        trust_remote_code=True  # 信任远程代码
    )
    model.to("mps")

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    instruction_text = f"<|im_start|>system\n现在进行实体抽取任务，类型只能是ORG（机构），PER（人名），LOC（地名），直接输出结果，格式为'类型：实体'，识别不出实体就输出'未识别到实体'<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
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

    model.print_trainable_parameters()

    return model

# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_ner_Qwen1.5",
        per_device_train_batch_size=6,  
        gradient_accumulation_steps=4, 
        logging_steps=100,   
        do_eval=True,  
        eval_steps=50,   
        num_train_epochs=5,  
        save_steps=50,   
        learning_rate=1e-4,
        gradient_checkpointing=True,  
        report_to="none" 
    )

# 预测函数
def predict_ner(model, tokenizer, text):
    """预测单个文本的实体"""
    # 构建对话消息列表
    messages = [
        {"role": "system", "content": f"现在进行实体抽取任务，类型只能是ORG（机构），PER（人名），LOC（地名），直接输出结果，格式为'类型：实体'，识别不出实体就输出'未识别到实体'"},  # 系统提示
        {"role": "user", "content": f"请从以下文本中抽取实体，包含机构（ORG），人名（PER），地名（LOC）: {text}"}   # 用户输入
    ]


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
            max_new_tokens=100,                  # 最多生成512个新token
            do_sample=False,                     # 关闭随机采样，生成确定性结果
            temperature=0.3,                     # 降低温度，减少重复
            top_p=0.95,                          # 保留概率最高的选项
            repetition_penalty=1.2,              # 对重复token惩罚
            pad_token_id=tokenizer.pad_token_id,  # 填充token的ID
            eos_token_id=tokenizer.eos_token_id   # 结束token的ID
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # 解析字符串为元组列表
    entities = []
    if response != "未识别到实体":
        for line in response.split("\n"):
            if ":" in line:
                entity_type, entity = line.split(":", 1)  # 按第一个冒号分割
                entities.append((entity.strip(), entity_type.strip()))  # 存为 (实体, 类型)
    return entities  # 返回元组列表


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图"""
    pred_labels = []

    # 遍历所有测试文本，显示进度条
    for text in tqdm(test_texts, desc="预测实体"):
        try:
            entities = predict_ner(model, tokenizer, text)
            pred_labels.append(
                {
                    "text": text,
                    "entities": entities
                }
            )
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    ds_train, ds_val = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "/Users/wangyingyue/materials/大模型学习资料——八斗/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    train_tokenized = ds_train.map(process_func_with_tokenizer, remove_columns=ds_train.column_names)
    val_tokenized = ds_val.map(process_func_with_tokenizer, remove_columns=ds_val.column_names)

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
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    model.save_pretrained("./output_ner_Qwen1.5/")  # 保存模型
    tokenizer.save_pretrained("./output_ner_Qwen1.5/")  # 保存分词器

def test_example():
    model_path = "/Users/wangyingyue/materials/大模型学习资料——八斗/models/Qwen/Qwen3-0.6B"
    tokenizer, base_model = initialize_model_and_tokenizer(model_path)

    adapter_path = "./output_ner_Qwen1.5/"
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

    # 测试预测
    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
        '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
        '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
        '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
    ]
    # 批量预测并打印结果
    results = batch_predict(model, tokenizer, test_sentences)
    for res in results:
        print(f"句子：{res['text']}")
        if res['entities']:
            for entity, entity_typ in res['entities']:
                print(f"  {entity_typ}: {entity}")
        else:
                print("未识别到实体")
        print()


if __name__ == "__main__":
    # 训练模型（首次运行时启用）
    # result_df = main()

    # 测试模型（训练完成后运行）
    test_example()
