import os.path
from typing import Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

from config import Config

if __name__ == '__main__':
    # 相对路径
    BERT_WEIGHT = os.path.join("..", Config["model_path"], Config["model_name"]).__str__()
else:
    BERT_WEIGHT = os.path.join(Config["model_path"], Config["model_name"]).__str__()

BERT_PATH = Config["bert_path"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=2)
model.load_state_dict(torch.load(BERT_WEIGHT, weights_only=True))
model.to(device)


class DataGenerator(Dataset):
    def __init__(self, text: list[str]):
        self.load(text)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load(self, text):
        self.data = []
        for t in text:
            t = tokenizer(t, truncation=True, padding='max_length', max_length=100, return_tensors='pt')
            input_ids = t['input_ids'].squeeze(0)
            attention_mask = t['attention_mask'].squeeze(0)
            self.data.append([input_ids.to(device), attention_mask.to(device)])


def model_for_bert(request_text: Union[str, list[str]]):
    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("只支持格式为字符串或字符串列表")

    dg = DataGenerator(request_text)
    dl = DataLoader(dg, batch_size=16, shuffle=False)
    model.eval()
    output = []
    for batch in dl:
        with torch.no_grad():
            input_ids, attention_mask = batch
            predict = model(input_ids, attention_mask=attention_mask)
        output.extend(torch.argmax(predict[0], dim=-1).cpu().numpy().tolist())
    if isinstance(request_text, str):
        classify_result = output[0]
    else:
        classify_result = output
    return classify_result


if __name__ == '__main__':
    text = "分量足，味道好，服务态度好"
    o = model_for_bert(text)
    print(o)
