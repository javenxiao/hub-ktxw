import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from config import BERT_MODEL_PKL_PATH,BERT_MODEL_PERTRAINED_PATH
import numpy as np
from typing import Union, List
from safetensors.torch import load_file

from config import CATEGORY_NAME

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PERTRAINED_PATH, num_labels=2)
model.load_state_dict(load_file(BERT_MODEL_PKL_PATH))

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")

    test_encodings = tokenizer(
        list(request_text),
        truncation=True,
        padding=True,
        max_length=30,
        return_tensors="pt"
    )

    input_ids = test_encodings['input_ids'].to(device)
    attention_mask = test_encodings['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    pred = np.argmax(logits.cpu().numpy(), axis=1)
    output = pred.tolist()
    output = [CATEGORY_NAME[x] for x in output]
    if len(output) == 1:
        return output[0]
    else:
        return output