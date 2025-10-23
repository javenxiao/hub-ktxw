import pandas as pd
import json
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
import os
import requests

# 设置离线模式，避免网络连接问题
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class QAModelTrainer:
    """问答模型训练器"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None

    def check_internet_connection(self):
        """检查网络连接"""
        try:
            requests.get("https://huggingface.co", timeout=5)
            return True
        except:
            return False

    def load_cmrc2018_data(self, data_path, max_samples=1000):
        """加载CMRC2018数据集"""
        try:
            # 尝试加载训练数据
            train_path = os.path.join(data_path, 'train.json')
            dev_path = os.path.join(data_path, 'dev.json')

            if os.path.exists(train_path):
                # 使用正确的编码方式读取JSON文件
                with open(train_path, 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
                with open(dev_path, 'r', encoding='utf-8') as f:
                    dev_data = json.load(f)
                print("成功加载CMRC2018数据集")
            else:
                # 如果文件不存在，使用示例数据
                print("CMRC2018数据集不存在，使用示例数据")
                return self.create_sample_data(max_samples)

            # 准备训练数据
            train_paragraphs, train_questions, train_answers = self.prepare_dataset(train_data, max_samples)
            val_paragraphs, val_questions, val_answers = self.prepare_dataset(dev_data, max_samples // 10)

            # 转换为指令格式
            train_formatted = self.convert_to_instruction_format(train_paragraphs, train_questions, train_answers)
            val_formatted = self.convert_to_instruction_format(val_paragraphs, val_questions, val_answers)

            # 创建数据集
            train_df = pd.DataFrame(train_formatted)
            val_df = pd.DataFrame(val_formatted)

            self.train_dataset = Dataset.from_pandas(train_df)
            self.eval_dataset = Dataset.from_pandas(val_df)

            print(f"训练集大小: {len(self.train_dataset)}")
            print(f"验证集大小: {len(self.eval_dataset)}")

            return self.train_dataset, self.eval_dataset

        except Exception as e:
            print(f"加载数据时出错: {e}")
            return self.create_sample_data(max_samples)

    def prepare_dataset(self, data, max_samples):
        """准备数据集"""
        paragraphs = []
        questions = []
        answers = []

        for paragraph in data['data'][:max_samples]:
            context = paragraph['paragraphs'][0]['context']
            for qa in paragraph['paragraphs'][0]['qas'][:5]:  # 每个段落最多取5个问题
                paragraphs.append(context)
                questions.append(qa['question'])
                if qa.get('answers') and len(qa['answers']) > 0:  # 确保有答案
                    answers.append({
                        'answer_start': [qa['answers'][0]['answer_start']],
                        'text': [qa['answers'][0]['text']]
                    })
                else:
                    answers.append({
                        'answer_start': [0],
                        'text': [""]
                    })

        return paragraphs, questions, answers

    def convert_to_instruction_format(self, paragraphs, questions, answers):
        """将问答数据转换为指令格式"""
        formatted_data = []

        for context, question, answer in zip(paragraphs, questions, answers):
            # 构建指令
            instruction = f"根据以下上下文回答问题。\n上下文：{context}\n问题：{question}"

            # 构建输出
            output = answer['text'][0] if answer['text'][0] else "根据上下文无法找到答案"

            formatted_data.append({
                "instruction": instruction,
                "input": "",
                "output": output
            })

        # 打印示例
        print("问答数据示例:")
        for i in range(min(2, len(formatted_data))):
            print(f"输入: {formatted_data[i]['instruction']}")
            print(f"输出: {formatted_data[i]['output']}")
            print("---")

        return formatted_data

    def create_sample_data(self, max_samples=100):
        """创建示例数据"""
        print("创建示例问答数据...")

        sample_data = [
            {
                "instruction": "根据以下上下文回答问题。\n上下文：北京是中国的首都，有着悠久的历史和丰富的文化遗产。\n问题：北京是哪个国家的首都？",
                "input": "",
                "output": "中国"
            },
            {
                "instruction": "根据以下上下文回答问题。\n上下文：长江是中国最长的河流，全长约6300公里，流经多个省份。\n问题：长江的长度是多少？",
                "input": "",
                "output": "约6300公里"
            },
            {
                "instruction": "根据以下上下文回答问题。\n上下文：人工智能是计算机科学的一个分支，旨在创造能够执行智能任务的机器。\n问题：人工智能是什么？",
                "input": "",
                "output": "计算机科学的一个分支，旨在创造能够执行智能任务的机器"
            },
            {
                "instruction": "根据以下上下文回答问题。\n上下文：太阳系有八大行星，按照离太阳的距离从近到远分别是：水星、金星、地球、火星、木星、土星、天王星和海王星。\n问题：太阳系有多少颗行星？",
                "input": "",
                "output": "八颗"
            }
        ]

        # 重复样本以增加数据量
        formatted_data = sample_data * (max_samples // len(sample_data) + 1)
        formatted_data = formatted_data[:max_samples]

        train_df = pd.DataFrame(formatted_data[:int(len(formatted_data) * 0.8)])
        val_df = pd.DataFrame(formatted_data[int(len(formatted_data) * 0.8):])

        self.train_dataset = Dataset.from_pandas(train_df)
        self.eval_dataset = Dataset.from_pandas(val_df)

        print(f"创建了 {len(self.train_dataset)} 条训练数据和 {len(self.eval_dataset)} 条验证数据")
        return self.train_dataset, self.eval_dataset

    def initialize_model_and_tokenizer(self):
        """初始化tokenizer和模型"""
        try:
            # 检查网络连接
            if not self.check_internet_connection():
                print("网络连接不可用，尝试使用本地模型...")
                # 尝试使用本地模型路径
                local_paths = ["../models/Qwen/Qwen3-0.6B/"]

                for local_path in local_paths:
                    if os.path.exists(local_path):
                        self.model_path = local_path
                        print(f"使用本地模型: {local_path}")
                        break
                else:
                    print("未找到本地模型，尝试使用较小的模型...")
                    # 使用一个更小的模型作为备用
                    self.model_path = "microsoft/DialoGPT-small"

            print(f"加载模型: {self.model_path}")

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=not self.check_internet_connection()  # 网络不可用时只使用本地文件
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float32,  # 使用float32提高稳定性
                trust_remote_code=True,
                local_files_only=not self.check_internet_connection()  # 网络不可用时只使用本地文件
            )

            print("模型和tokenizer初始化成功")
            return self.tokenizer, self.model

        except Exception as e:
            print(f"加载模型失败: {e}")
            # 尝试使用更简单的模型
            try:
                from transformers import GPT2Tokenizer, GPT2LMHeadModel
                print("尝试使用GPT2模型...")
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.model = GPT2LMHeadModel.from_pretrained("gpt2")

                # 设置pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                print("使用GPT2模型成功")
                return self.tokenizer, self.model
            except Exception as e2:
                print(f"备用模型也加载失败: {e2}")
                raise

    def process_func(self, example, max_length=384):  # 减少最大长度以节省内存
        """
        处理单个样本的函数
        将指令和输出转换为模型训练格式
        """
        try:
            # 更简单的提示词格式
            prompt = f"用户: {example['instruction']}\n助手: {example['output']}"

            # Tokenize整个文本
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )

            # 简单的标签设置（所有token都参与训练）
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized
        except Exception as e:
            print(f"处理数据时出错: {e}")
            return {
                "input_ids": [self.tokenizer.pad_token_id],
                "attention_mask": [1],
                "labels": [self.tokenizer.pad_token_id]
            }

    def setup_lora(self):
        """设置LoRA配置并应用到模型"""
        try:
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 减少目标模块
                inference_mode=False,
                r=4,  # 减少r值
                lora_alpha=16,  # 减少alpha值
                lora_dropout=0.05  # 减少dropout
            )

            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()

            return self.model
        except Exception as e:
            print(f"LoRA设置失败: {e}")
            print("使用全参数微调")
            return self.model

    def setup_training_args(self):
        """设置训练参数"""
        return TrainingArguments(
            output_dir="./output_Qwen_QA",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,  # 减少梯度累积步数
            logging_steps=10,
            num_train_epochs=2,  # 减少训练轮数
            save_steps=100,
            learning_rate=1e-4,  # 降低学习率
            warmup_steps=20,
            eval_strategy="no",  # 关闭评估以避免网络问题
            save_total_limit=1,
            load_best_model_at_end=False,  # 关闭最佳模型加载
            report_to="none",
            dataloader_pin_memory=False,
            no_cuda=not torch.cuda.is_available(),  # 根据CUDA可用性设置
        )

    def train(self):
        """训练模型"""
        try:
            # 1. 加载数据
            print("加载问答数据...")
            data_path = "./cmrc2018_public"
            self.load_cmrc2018_data(data_path, max_samples=200)  # 减少样本数量

            # 2. 初始化模型和tokenizer
            print("初始化模型和tokenizer...")
            self.initialize_model_and_tokenizer()

            # 3. 处理数据
            print("处理训练数据...")
            process_func_with_tokenizer = lambda example: self.process_func(example)
            tokenized_train = self.train_dataset.map(process_func_with_tokenizer,
                                                     remove_columns=self.train_dataset.column_names)
            tokenized_eval = self.eval_dataset.map(process_func_with_tokenizer,
                                                   remove_columns=self.eval_dataset.column_names)

            print(f"训练集大小: {len(tokenized_train)}")
            print(f"验证集大小: {len(tokenized_eval)}")

            # 4. 设置LoRA
            print("设置LoRA...")
            self.model.enable_input_require_grads()
            self.model = self.setup_lora()

            # 5. 配置训练参数
            print("配置训练参数...")
            training_args = self.setup_training_args()

            # 6. 创建Trainer并开始训练
            print("开始训练...")
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_train,
                data_collator=DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer,
                    padding=True
                ),
            )

            trainer.train()

            # 7. 保存模型（简化保存过程）
            print("保存模型...")
            try:
                # 只保存必要的文件
                self.model.save_pretrained("./output_Qwen_QA", safe_serialization=False)
                self.tokenizer.save_pretrained("./output_Qwen_QA")
                print("模型保存成功")
            except Exception as e:
                print(f"模型保存失败: {e}")
                print("但训练已完成，可以继续测试")

            print("训练完成!")
            return self.model, self.tokenizer

        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict_answer(self, context, question, device='cpu'):
        """预测答案"""
        if self.model is None or self.tokenizer is None:
            print("模型未加载")
            return ""

        try:
            # 将模型移动到指定设备
            self.model.to(device)

            # 更简单的提示词格式
            prompt = f"根据以下上下文回答问题：{context} 问题：{question} 答案："

            # Tokenize输入
            inputs = self.tokenizer([prompt], return_tensors="pt").to(device)

            # 生成预测
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )

            # 提取生成的文本（去掉输入部分）
            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response.strip()

        except Exception as e:
            print(f"预测时出错: {e}")
            return ""


# 测试函数
def test_qa_model(trainer, device='cpu'):
    """测试问答模型"""
    if trainer.model is None or trainer.tokenizer is None:
        print("模型未正确加载，无法测试")
        return

    # 测试用例
    test_cases = [
        {
            "context": "北京是中国的首都，有着悠久的历史和丰富的文化遗产。故宫、天安门和长城都是北京的著名景点。",
            "question": "北京是哪个国家的首都？"
        },
        {
            "context": "长江是中国最长的河流，全长约6300公里，流经多个省份，最终注入东海。",
            "question": "长江的长度是多少？"
        }
    ]

    print("\n=== 问答模型测试 ===")
    for i, test_case in enumerate(test_cases):
        context = test_case["context"]
        question = test_case["question"]

        answer = trainer.predict_answer(context, question, device)

        print(f"测试 {i + 1}:")
        print(f"上下文: {context}")
        print(f"问题: {question}")
        print(f"模型回答: {answer}")
        print()


# 主函数
def main():
    """主执行函数"""
    # 初始化训练器
    model_path = "../models/Qwen/Qwen3-0.6B/"  # 可以替换为你的本地模型路径
    trainer = QAModelTrainer(model_path)

    # 训练模型
    model, tokenizer = trainer.train()

    # 测试模型
    if model is not None:
        trainer.model = model
        trainer.tokenizer = tokenizer
        test_qa_model(trainer, device)
    else:
        print("模型训练失败，跳过测试")


if __name__ == "__main__":
    main()
