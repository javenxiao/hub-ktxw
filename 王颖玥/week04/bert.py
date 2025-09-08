import torch
import numpy as np
import traceback

from typing import Union, List
from fastapi import FastAPI
from transformers import AutoTokenizer, BertForSequenceClassification

from data_schema import TextClassifyResponse
from data_schema import TextClassifyRequest

app = FastAPI(title="外卖评价分类")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./models/")
model = BertForSequenceClassification.from_pretrained("./models/", num_labels=2)

model.load_state_dict(torch.load("./bert_model.pt"))
model.to(device)

def predict(request_text: Union[str, List[str]]) -> List[int]:
    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")

    test_encoding = tokenizer(
        request_text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    )
    test_encoding = {k: v.to(device) for k, v in test_encoding.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**test_encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions.tolist()

@app.post("/predict")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    response = TextClassifyResponse(
        request_text=req.request_text,
        classify_result=[],
        classify_meaning=[],
        error_msg=""
    )

    try:
        pred = predict(req.request_text)
        response.classify_result = pred
        response.classify_meaning = ["好评" if x == 1 else "差评" for x in pred]
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = []
        response.classify_meaning = []
        response.error_msg = traceback.format_exc()

    return response
