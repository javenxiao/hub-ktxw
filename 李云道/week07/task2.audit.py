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
    作业批改：
        提示词设计合理，避免使用BIO标签输出（LLM不理解且tokenize后非整体）
        应该是：
            请帮我识别下面文本的实体，并用“词-标签”的格式输出。
            输入文本：我明天要去北京
            输出文本：北京-地区
'''
