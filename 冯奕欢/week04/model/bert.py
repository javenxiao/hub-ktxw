import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from config import MODEL_PATH, WEIGHT_PATH


# 定义数据集
class ClassificationDataset(Dataset):

    def __init__(self, x_encoding, y):
        self.x_encoding = x_encoding
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        items = {key: torch.tensor(value[idx]) for key, value in self.x_encoding.items()}
        items['labels'] = torch.tensor(self.y[idx])
        return items


# 分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# 模型
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
# 加载参数
model.load_state_dict(torch.load(WEIGHT_PATH))


def bert_classify(text) -> int:
    """
    使用微调的BERT模型预测文本分类
    :param text: 输入文案
    :return 返回分类标签
    """
    text_encoding = tokenizer(list(text), padding=True, truncation=True, max_length=64)
    labels = [0]
    dataset = ClassificationDataset(text_encoding, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    predict_results = []
    with torch.no_grad():
        for items in dataloader:
            input_ids = items['input_ids']
            attention_mask = items['attention_mask']
            labels = items['labels']
            # 前向传播 计算结果
            model_result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = model_result.logits
            predicts = logits.detach().numpy()
            predict_result = np.argmax(predicts, axis=1).flatten()[0]
            predict_results.append(predict_result)
    return int(predict_results[0])
