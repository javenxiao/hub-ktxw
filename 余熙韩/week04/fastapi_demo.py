import torch

from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional

import sys
sys.path.append('../')

from utils.device_utils import get_device_info




class TextClassifyRequest(BaseModel):
    """
    请求格式
    """
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")



class SentimentAnalyzer:
    """外卖评论情感分析器"""

    def __init__(self, model_path, tokenizer_path):
        """
        初始化情感分析器
        
        Args:
            model_path: 训练好的模型路径
            tokenizer_path: 分词器路径
        """
        # 获取设备
        self.device, self.device_type, _ = get_device_info()
        print(f"🎯 加载模型到设备: {self.device_type}")

        # 加载分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

    def predict_single(self, text):
        """
        预测单条文本的情感
        
        Args:
            text: 输入的评论文本
            
        Returns:
            dict: 包含预测结果和置信度
        """
        # 文本预处理
        inputs = self.tokenizer(text,
                                truncation=True,
                                padding=True,
                                max_length=128,
                                return_tensors="pt").to(self.device)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)

            # 获取预测结果
            # confidence = predictions.max().item()
            type = predictions.data.max(1, keepdim=True)[1].item()
        return {
            'text': text,
            'type': f"{type}",
        }

    def predict_batch(self, texts):
        """
        批量预测多条文本
        
        Args:
            texts: 文本列表
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for text in texts:
            results.append(self.predict_single(text))
        return results


model_config = dict(best_model_path="./results/checkpoint-2400",tokenizer_path="../models/google-bert/bert-base-chinese")

model = SentimentAnalyzer(model_config['best_model_path'], model_config['tokenizer_path'])


# 创建 FastAPI 应用实例
app = FastAPI()

# 定义根路径的 GET 请求处理函数
@app.get("/")
async def read_root():
    return {"Hello": "World"}

# 定义一个 POST 请求处理函数，用于接收文本并返回预测结果
@app.post("/predict")
async def predict(request: TextClassifyRequest):
    response = TextClassifyResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        classify_result=model.predict_single(request.request_text).get("type"),
        classify_time=0,
        error_msg="",
    )
    return response


# 定义一个 POST 请求处理函数，用于接收文本列表并返回批量预测结果
@app.post("/predict_batch")
async def predict_batch(request: TextClassifyRequest):
    response = TextClassifyResponse(
        request_id=request.request_id,
        request_text=request.request_text,
        classify_result=[item.get("type") for item in model.predict_batch(request.request_text)],
        classify_time=0,
        error_msg="",
    )
    return response



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
