from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained('/assets/weights/bert.pt')
tokenizer = BertTokenizer.from_pretrained('/assets/weights/bert.pt')
model.eval()

# 定义FastAPI应用
app = FastAPI()


class ReviewRequest(BaseModel):
    review: str

class ReviewResponse(BaseModel):
    label: int
    confidence: float

# 定义分类接口
@app.post("/classify", response_model=ReviewResponse)
async def classify_review(request: ReviewRequest):
    inputs = tokenizer(request.review, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()
    return ReviewResponse(label=label, confidence=confidence)


