import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    PeftModel,
    PeftConfig
)
from datasets import Dataset
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据（使用与05_BERT知识问答.py相同的数据集）
def load_data(file_path, max_samples=1000):
    data = json.load(open(file_path))
    
    paragraphs = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            if len(paragraphs) >= max_samples:
                break
            paragraphs.append(context)
            questions.append(qa['question'])
            answers.append({
                'answer_start': [qa['answers'][0]['answer_start']],
                'text': [qa['answers'][0]['text']]
            })
        if len(paragraphs) >= max_samples:
            break

    return paragraphs, questions, answers

# 加载训练和验证数据
train_paragraphs, train_questions, train_answers = load_data('./cmrc2018_public/train.json', 1000)
val_paragraphs, val_questions, val_answers = load_data('./cmrc2018_public/dev.json', 100)

print(f"训练样本数: {len(train_paragraphs)}")
print(f"验证样本数: {len(val_paragraphs)}")

# 初始化Qwen tokenizer和模型
model_name = "Qwen/Qwen2.5-1.5B"  # 可以选择不同大小的Qwen模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 如果tokenizer没有pad_token，设置为eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建数据集字典
train_dataset_dict = {
    'context': train_paragraphs,
    'question': train_questions,
    'answers': train_answers
}

val_dataset_dict = {
    'context': val_paragraphs,
    'question': val_questions,
    'answers': val_answers
}

# 转换为Hugging Face Dataset
train_dataset = Dataset.from_dict(train_dataset_dict)
val_dataset = Dataset.from_dict(val_dataset_dict)

# 预处理函数 - 适配Qwen模型
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # Tokenize - 针对Qwen模型调整参数
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",  # 只截断context部分
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 由于可能有溢出，需要重新映射样本
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 获取对应的原始样本
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]

        # 如果没有答案，设置默认值
        if len(answer["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # 找到token的起始和结束位置
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 找到context的开始和结束
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # 如果答案完全在context之外，标记为不可回答
        if (context_start >= len(offsets) or context_end >= len(offsets) or
            offsets[context_start][0] > end_char or offsets[context_end][1] < start_char):
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # 否则找到答案的token位置
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

            # 确保位置在有效范围内
            start_position = max(context_start, min(start_position, context_end))
            end_position = max(context_start, min(end_position, context_end))
            
            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples

# 应用预处理
print("预处理训练数据...")
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print("预处理验证数据...")
tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

print("数据预处理完成")

# 加载Qwen问答模型
model = AutoModelForQuestionAnswering.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 配置LoRA参数 - 针对问答任务优化
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # 问答任务可以视为序列分类
    inference_mode=False,
    r=16,  # LoRA秩
    lora_alpha=32,  # LoRA alpha参数
    lora_dropout=0.1,  # LoRA dropout
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 将模型移动到设备
model.to(device)

# 设置训练参数（针对问答任务优化）
training_args = TrainingArguments(
    output_dir='./qwen-lora-qa-model',
    learning_rate=2e-4,  # 问答任务使用稍大的学习率
    per_device_train_batch_size=2,  # 问答任务需要更多内存，减小batch_size
    per_device_eval_batch_size=2,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    gradient_accumulation_steps=4,  # 累积梯度以补偿小的batch_size
    report_to="none",
)

# 数据收集器
data_collator = DefaultDataCollator()

# 定义计算指标的函数 - 问答任务专用
def compute_metrics(p):
    start_logits, end_logits = p.predictions
    start_positions = p.label_ids[:, 0]
    end_positions = p.label_ids[:, 1]
    
    # 计算起始位置准确率
    start_preds = np.argmax(start_logits, axis=1)
    start_accuracy = np.mean(start_preds == start_positions)
    
    # 计算结束位置准确率
    end_preds = np.argmax(end_logits, axis=1)
    end_accuracy = np.mean(end_preds == end_positions)
    
    # 计算完全匹配准确率（起始和结束位置都正确）
    exact_match = np.mean((start_preds == start_positions) & (end_preds == end_positions))
    
    return {
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "exact_match": exact_match,
        "overall_accuracy": (start_accuracy + end_accuracy) / 2
    }

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
print("开始Qwen-LoRA问答模型微调训练...")
trainer.train()

# 保存模型
trainer.save_model()
print("模型保存完成")

# 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 加载最佳模型进行预测
try:
    # 尝试加载最佳模型
    config = PeftConfig.from_pretrained('./qwen-lora-qa-model')
    best_model = AutoModelForQuestionAnswering.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    best_model = PeftModel.from_pretrained(best_model, './qwen-lora-qa-model')
    best_model.to(device)
    print("成功加载最佳模型")
except Exception as e:
    print(f"加载最佳模型失败: {e}，使用当前模型")
    best_model = model

# 改进的预测函数
def predict_answer(context: str, question: str, model, tokenizer, max_length: int = 512) -> str:
    """
    使用模型预测答案
    """
    model.eval()
    
    # Tokenize输入
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
        stride=128,
        return_offsets_mapping=True
    )
    
    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    offset_mapping = inputs.pop('offset_mapping').cpu().numpy()[0]
    
    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取预测的起始和结束位置
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()
    
    # 找到最可能的答案跨度
    start_idx = np.argmax(start_logits, axis=1)[0]
    end_idx = np.argmax(end_logits, axis=1)[0]
    
    # 确保结束位置不小于起始位置
    if end_idx < start_idx:
        end_idx = start_idx
    
    # 将token位置转换为字符位置
    start_char = offset_mapping[start_idx][0]
    end_char = offset_mapping[end_idx][1]
    
    # 提取答案文本
    answer = context[start_char:end_char]
    
    # 清理答案
    answer = answer.strip()
    
    # 如果答案为空或过长，返回空字符串
    if not answer or len(answer) > 100:
        return ""
    
    return answer

# 更智能的预测函数（带后处理）
def smart_predict(context: str, question: str, model, tokenizer) -> Dict:
    """
    智能预测答案，包含多种策略
    """
    # 策略1: 直接预测
    answer1 = predict_answer(context, question, model, tokenizer)
    
    # 策略2: 调整最大长度重新预测
    if not answer1 or len(answer1) < 2:
        answer2 = predict_answer(context, question, model, tokenizer, max_length=256)
    else:
        answer2 = answer1
    
    # 选择最合理的答案
    final_answer = answer1 if answer1 else answer2
    
    # 验证答案是否在上下文中
    if final_answer and final_answer not in context:
        # 如果答案不在上下文中，尝试找到最相似的片段
        words = final_answer.replace(' ', '')
        for i in range(len(context) - len(words) + 1):
            if context[i:i+len(words)] == words:
                final_answer = context[i:i+len(words)]
                break
        else:
            final_answer = ""  # 如果找不到，返回空
    
    return {
        "answer": final_answer,
        "context": context,
        "question": question
    }

# 测试预测
print("\n问答测试结果:")

# 测试用例
test_cases = [
    {
        "context": "北京是中国的首都，有着悠久的历史和丰富的文化遗产。故宫位于北京市中心，是天安门广场北侧的一座古代建筑群。",
        "question": "北京是哪个国家的首都？"
    },
    {
        "context": "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。深度学习是人工智能的一个子领域。",
        "question": "人工智能是什么？"
    },
    {
        "context": "黄河是中国第二长河，全长约5464公里，发源于青海省，流经九个省区，最终注入渤海。",
        "question": "黄河的长度是多少？"
    }
]

# 添加验证集中的样本进行测试
for i in range(min(3, len(val_paragraphs))):
    test_cases.append({
        "context": val_paragraphs[i],
        "question": val_questions[i]
    })

for i, test_case in enumerate(test_cases):
    try:
        result = smart_predict(
            test_case["context"], 
            test_case["question"], 
            best_model, 
            tokenizer
        )
        
        print(f"\n测试案例 {i + 1}:")
        print(f"问题: {result['question']}")
        print(f"预测答案: {result['answer']}")
        
        # 如果是验证集样本，显示真实答案
        if i >= 3 and i - 3 < len(val_answers):
            true_answer = val_answers[i - 3]['text'][0]
            print(f"真实答案: {true_answer}")
            print(f"匹配: {result['answer'] == true_answer}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"处理测试案例 {i + 1} 时出错: {e}")

# 批量评估函数
def evaluate_model(model, tokenizer, contexts, questions, answers, num_samples=10):
    """
    批量评估模型性能
    """
    correct = 0
    total = min(num_samples, len(contexts))
    
    print(f"\n批量评估结果（{total}个样本）:")
    
    for i in range(total):
        try:
            result = smart_predict(contexts[i], questions[i], model, tokenizer)
            predicted_answer = result["answer"]
            true_answer = answers[i]['text'][0]
            
            # 简单的字符串匹配（可以改进为更复杂的匹配逻辑）
            match = predicted_answer == true_answer or true_answer in predicted_answer or predicted_answer in true_answer
            
            if match:
                correct += 1
            
            print(f"{i+1}. 问题: {questions[i][:50]}...")
            print(f"   预测: {predicted_answer}")
            print(f"   真实: {true_answer}")
            print(f"   结果: {'✓' if match else '✗'}")
            print()
            
        except Exception as e:
            print(f"评估样本 {i+1} 时出错: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    print(f"准确率: {accuracy:.2%} ({correct}/{total})")
    
    return accuracy

# 执行批量评估
evaluate_model(best_model, tokenizer, val_paragraphs, val_questions, val_answers, 10)

# 打印训练总结
print("\n训练总结:")
print(f"模型: {model_name}")
print(f"微调方法: LoRA (r={lora_config.r}, alpha={lora_config.lora_alpha})")
print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"训练数据量: {len(train_paragraphs)}")
print(f"验证数据量: {len(val_paragraphs)}")

