from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, BertForSequenceClassification
import numpy as np
import torch
import time
import traceback
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型路径 - 根据你的实际情况调整
BERT_MODEL_PKL_PATH = "./bert.pt"
BERT_MODEL_PRETRAINED_PATH = "./models/bert-base-chinese"

CATEGORY_NAME = ['0', '1']

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PRETRAINED_PATH)

# 加载模型
try:
    # 首先加载预训练模型结构
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_PRETRAINED_PATH,
        num_labels=2  # 确保与训练时的类别数一致
    )

    # 然后加载训练好的权重
    model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    logger.info("模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise e

app = FastAPI(title="BERT文本分类API", version="1.0.0")


class TextClassifyRequest(BaseModel):
    """
    请求格式
    """
    request_id: Optional[str] = Field(None, description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(None, description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field("", description="异常信息")


def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """使用BERT模型进行文本分类"""
    if isinstance(request_text, str):
        texts = [request_text]
        return_single = True
    elif isinstance(request_text, list):
        texts = request_text
        return_single = False
    else:
        raise ValueError("输入格式不支持，应为字符串或字符串列表")

    # 编码文本
    try:
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=64,  # 与训练时保持一致
            return_tensors="pt"
        )
    except Exception as e:
        logger.error(f"文本编码失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"文本编码失败: {str(e)}")

    # 移动到设备
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # 预测
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.cpu().numpy().tolist()
    except Exception as e:
        logger.error(f"模型预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型预测失败: {str(e)}")

    # 转换为类别名称
    results = [CATEGORY_NAME[pred] for pred in predictions]

    # 如果是单条文本，返回单个结果
    if return_single:
        return results[0]
    else:
        return results


@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用BERT进行文本分类

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "成功"
    except Exception as err:
        error_trace = traceback.format_exc()
        logger.error(f"分类失败: {error_trace}")
        response.error_msg = f"分类失败: {str(err)}"
        # 可以在这里决定是否抛出HTTP异常
        # raise HTTPException(status_code=500, detail=response.error_msg)

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
