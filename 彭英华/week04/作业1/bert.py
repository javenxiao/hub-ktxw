import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from typing import Union
from transformers import AutoTokenizer,BertForSequenceClassification,BertTokenizer,AutoModelForSequenceClassification
data = pd.read_csv("./dataset/waimai_10k.csv")
# labels = [val for val in data['label']]
classify_list=["负面评价","正面评价"]

class Newdataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self,idx):
        item = {key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
model_path = "./models/weights/bert.pt"
tokenizer = BertTokenizer.from_pretrained("../models/google-bert/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("../models/google-bert/bert-base-chinese",num_labels=2)
model.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.cuda()
def model_for_bert(request_text:Union[str,list[str]]) -> Union[str,list[str]]:
    if isinstance(request_text,str):
        request_text = [request_text]
    elif isinstance(request_text,list):
        pass
    else:
        raise Exception("格式不支持")
    test_encoding = tokenizer(list(request_text),truncation=True,max_length=64,padding=True)
    test_dataset = Newdataset(test_encoding,[0]*len(request_text))
    test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=False)
    pred = []
    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch['labels'].cuda()
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        pred +=list(np.argmax(logits,axis=-1).flatten())
    classify_result = [classify_list[i] for i in pred]
    return classify_result
# dataset = data['review'].values
# labels = data['label'].values
# train_data,test_data,train_labels,test_labels = train_test_split(dataset,labels,test_size=0.2,stratify=labels)
# for i in zip(test_data,test_labels):
#     if model_for_bert(i[0])[0]==i[1]:
#         print(i[0],i[1])
