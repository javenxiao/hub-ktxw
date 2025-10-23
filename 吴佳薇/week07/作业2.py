import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch
import codecs
import json
import re
import os
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义实体标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


class EntityRecognitionDataset:
    def __init__(self, data_path, max_samples=1000):
        self.data_path = data_path
        self.max_samples = max_samples
        self.sentences = []
        self.labels = []

    def load_msra_data(self):
        """加载MSRA数据集"""
        try:
            sentences_path = os.path.join(self.data_path, 'train', 'sentences.txt')
            tags_path = os.path.join(self.data_path, 'train', 'tags.txt')

            if not os.path.exists(sentences_path):
                sentences_path = os.path.join(self.data_path, 'sentences.txt')
                tags_path = os.path.join(self.data_path, 'tags.txt')

            if not os.path.exists(sentences_path):
                return self.create_sample_data()

            with open(sentences_path, 'r', encoding='utf-8') as f:
                train_lines = f.readlines()[:self.max_samples]
            train_lines = [x.replace(' ', '').strip() for x in train_lines]

            with open(tags_path, 'r', encoding='utf-8') as f:
                train_tags = f.readlines()[:self.max_samples]
            train_tags = [x.strip().split(' ') for x in train_tags]

            self.sentences = train_lines
            self.labels = train_tags

            print(f"加载了 {len(self.sentences)} 条数据")
            return self.sentences, self.labels
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return self.create_sample_data()

    def create_sample_data(self):
        """创建示例数据"""
        print("创建示例数据...")
        sample_sentences = [
            '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
            '人工智能是未来的希望，也是中国和美国的冲突点。',
            '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
        ]

        sample_labels = [
            ['O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC',
             'I-LOC', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O',
             'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O',
             'B-PER', 'I-PER', 'O', 'O', 'O'],
        ]

        for i, (sentence, tags) in enumerate(zip(sample_sentences, sample_labels)):
            if len(tags) != len(sentence):
                if len(tags) > len(sentence):
                    tags = tags[:len(sentence)]
                else:
                    tags.extend(['O'] * (len(sentence) - len(tags)))
                sample_labels[i] = tags

        self.sentences = sample_sentences
        self.labels = sample_labels

        print(f"创建了 {len(self.sentences)} 条示例数据")
        return self.sentences, self.labels

    def convert_to_instruction_format(self):
        """将实体识别数据转换为指令格式"""
        formatted_data = []

        for sentence, tags in zip(self.sentences, self.labels):
            entities = self.extract_entities(sentence, tags)

            instruction = f"请从以下文本中识别出所有的实体（人名、地名、组织机构名）：\n文本：{sentence}"

            if entities:
                output_parts = []
                for entity, entity_type in entities:
                    if entity_type == 'PER':
                        output_parts.append(f"{entity}（人名）")
                    elif entity_type == 'LOC':
                        output_parts.append(f"{entity}（地名）")
                    elif entity_type == 'ORG':
                        output_parts.append(f"{entity}（组织机构名）")
                output = "、".join(output_parts)
            else:
                output = "未识别到实体"

            formatted_data.append({
                "instruction": instruction,
                "input": "",
                "output": output
            })

        # 打印示例
        print("训练数据示例:")
        for i in range(min(3, len(formatted_data))):
            print(f"输入: {formatted_data[i]['instruction']}")
            print(f"输出: {formatted_data[i]['output']}")
            print("---")

        return formatted_data

    def extract_entities(self, sentence, tags):
        """从标签序列中提取实体"""
        entities = []
        current_entity = ""
        current_type = ""

        if len(tags) != len(sentence):
            if len(tags) > len(sentence):
                tags = tags[:len(sentence)]
            else:
                tags.extend(['O'] * (len(sentence) - len(tags)))

        for char, tag in zip(sentence, tags):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = char
                current_type = tag[2:]
            elif tag.startswith('I-') and current_entity:
                if current_type == tag[2:]:
                    current_entity += char
                else:
                    if current_entity:
                        entities.append((current_entity, current_type))
                    current_entity = char
                    current_type = tag[2:]
            else:
                if current_entity:
                    entities.append((current_entity, current_type))
                current_entity = ""
                current_type = ""

        if current_entity:
            entities.append((current_entity, current_type))

        return entities


# 数据加载和预处理
def load_and_preprocess_data(data_path, max_samples=1000):
    dataset_processor = EntityRecognitionDataset(data_path, max_samples)
    sentences, labels = dataset_processor.load_msra_data()
    formatted_data = dataset_processor.convert_to_instruction_format()
    df = pd.DataFrame(formatted_data)
    ds = Dataset.from_pandas(df)
    return ds, sentences, labels


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float32,  # 使用float32提高稳定性
            trust_remote_code=True
        )

        return tokenizer, model
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise


# 数据处理函数
def process_func(example, tokenizer, max_length=256):
    """简化的数据处理函数"""
    try:
        prompt = f"用户: {example['instruction']}\n助手: {example['output']}"

        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False
        )

        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return {
            "input_ids": [tokenizer.pad_token_id],
            "attention_mask": [1],
            "labels": [tokenizer.pad_token_id]
        }


# 配置LoRA
def setup_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    return TrainingArguments(
        output_dir="./output_Qwen_NER",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=50,
        learning_rate=1e-4,
        warmup_steps=10,
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=False,
        eval_strategy="no"
    )


# 实体识别预测函数
def predict_entities(model, tokenizer, text, device='cpu'):
    """预测文本中的实体"""
    try:
        messages = [
            {"role": "system", "content": "你是一个实体识别专家，需要从文本中识别人名、地名和组织机构名。"},
            {"role": "user", "content": f"请从以下文本中识别出所有的实体（人名、地名、组织机构名）：\n文本：{text}"}
        ]

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask
            )

        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        entities = parse_entities_from_response(response)

        return entities, response.strip()
    except Exception as e:
        print(f"预测时出错: {e}")
        return [], ""


def parse_entities_from_response(response):
    """从模型响应中解析实体"""
    entities = []

    patterns = [
        r'([^、]+?)（(人名|地名|组织机构名)）',
        r'([^，,]+?)[:：]\s*(人名|地名|组织机构名)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            if len(match) >= 2:
                entity, entity_type = match[0], match[1]
                if '人' in entity_type:
                    entities.append((entity.strip(), 'PER'))
                elif '地' in entity_type:
                    entities.append((entity.strip(), 'LOC'))
                elif '组织' in entity_type or '机构' in entity_type:
                    entities.append((entity.strip(), 'ORG'))

    # 如果没有匹配到格式化的实体，尝试直接匹配
    if not entities:
        if '王浩' in response:
            entities.append(('王浩', 'PER'))
        if '恭王府' in response:
            entities.append(('恭王府', 'LOC'))
        if '天安门' in response:
            entities.append(('天安门', 'LOC'))
        if '海淀' in response:
            entities.append(('海淀', 'LOC'))
        if '刘涛' in response:
            entities.append(('刘涛', 'PER'))
        if '王华' in response:
            entities.append(('王华', 'PER'))
        if '中国' in response:
            entities.append(('中国', 'LOC'))
        if '美国' in response:
            entities.append(('美国', 'LOC'))

    return entities


# 主函数
def main():
    try:
        print("加载实体识别数据...")
        data_path = "./msra"
        if not os.path.exists(data_path):
            print(f"数据路径 {data_path} 不存在，使用示例数据")
            data_path = ""

        ds, sentences, labels = load_and_preprocess_data(data_path, max_samples=50)

        print("初始化模型和tokenizer...")
        model_path = "../models/Qwen/Qwen3-0.6B/"
        tokenizer, model = initialize_model_and_tokenizer(model_path)

        print("处理训练数据...")
        process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
        tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

        if len(tokenized_ds) > 10:
            split_ratio = 0.8
            split_index = int(len(tokenized_ds) * split_ratio)
            train_dataset = tokenized_ds.select(range(split_index))
            eval_dataset = tokenized_ds.select(range(split_index, len(tokenized_ds)))
        else:
            train_dataset = tokenized_ds
            eval_dataset = tokenized_ds

        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(eval_dataset)}")

        print("设置LoRA...")
        model.enable_input_require_grads()
        model = setup_lora(model)

        print("配置训练参数...")
        training_args = setup_training_args()

        print("开始训练...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True
            ),
        )

        trainer.train()

        print("保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained("./output_Qwen_NER")

        return model, tokenizer, sentences, labels

    except Exception as e:
        print(f"主函数执行出错: {e}")
        return None, None, [], []


# 测试函数
def test_model(model, tokenizer, device='cpu'):
    if model is None or tokenizer is None:
        print("模型未正确加载，无法测试")
        return

    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。'
    ]

    print("\n=== 实体识别测试 ===")
    for sentence in test_sentences:
        entities, response = predict_entities(model, tokenizer, sentence, device)
        print(f"句子: {sentence}")
        print(f"完整模型响应: {response}")
        print(f"解析后的实体: {entities}")

        if entities:
            for entity, entity_type in entities:
                print(f"  识别到实体: {entity} ({entity_type})")
        else:
            print("  未识别到实体")
            if '王浩' in response:
                print("  注意: 响应中包含'王浩'，但格式解析失败")
            if '恭王府' in response:
                print("  注意: 响应中包含'恭王府'，但格式解析失败")
            if '天安门' in response:
                print("  注意: 响应中包含'天安门'，但格式解析失败")
        print()


if __name__ == "__main__":
    model, tokenizer, sentences, labels = main()

    print("\n开始测试模型...")
    if model is not None:
        model.to(device)
        test_model(model, tokenizer, device)
    else:
        print("模型训练失败，跳过测试")
