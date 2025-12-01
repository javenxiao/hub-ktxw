from fastmcp import FastMCP
from typing import Union,Annotated
import requests
from snownlp import SnowNLP
from openai import OpenAI
mcp = FastMCP("My mcp Server")
import json
@mcp.tool
def text_sentiment_analysis(text:Annotated[str,"需要进行情感分析的文本"])->str:
    """调用大模型进行文本情感分析"""
    try:
      model = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                     api_key="sk-90aa2b7df82745f3a46373cc0ddd0497")
      messages = [{"role": "system", "content": "你是一个文本情感分析专家"},
                  {"role": "user", "content": f"""
                    请对下面的文本进行情感分析，并将其分类为积极，消极，中性三种结果,
                    {text}"""}]
      response = model.chat.completions.create(model="qwen-plus", messages=messages)
      return response.choices[0].message.content
    except Exception as e:
      return "无法进行文本情感分析"
@mcp.tool
def text_snownlp(text:Annotated[str,"需要进行情感分析的文本"]):
    """使用snownlp库进行文本情感分析"""
    s = SnowNLP(text)
    sentiment_score = s.sentiments
    if sentiment_score > 0.5:
        result=("这是一条正面评价")
    elif sentiment_score==0.5:
        result=("这是一条中性评价")
    else:
        result=("这是一条负面评价")
    return result
@mcp.tool
def text_baiduyun(text:Annotated[str,"需要进行情感分析的文本"]):
    """使用百度云的文本情感分析接口进行文本情感分析"""
    # 使用获取到的Access Token
    access_token = "24.52b62f6e33d2a5878f8e2a5c8b301139.2592000.1766114214.282335-120906486"
    headers = {'Content-Type': 'application/json'}
    payload = {
        'text': text
    }
    # 情感分析API的URL
    emotion_url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify'
    params = {'access_token': access_token}
    response = requests.post(emotion_url, params=params, data=json.dumps(payload), headers=headers)
    if response:
        analysis_result = response.json()
        # 输出分析结果
        positive_score = analysis_result['items'][0]['positive_prob']
        negative_score = analysis_result['items'][0]['negative_prob']
        confidence = analysis_result['items'][0]['confidence']
        if confidence<0.5:
            return "无法进行情感分析"
        else:
            if positive_score>negative_score:
                return "这是积极的情感"
            elif positive_score<negative_score:
                return "这是消极的情感"
            else:
                return "这是中性的情感"
    else:
        return "无法进行情感分析"
