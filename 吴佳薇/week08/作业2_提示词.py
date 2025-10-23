from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import json
import re
import time
from typing import Dict, List

# 初始化 FastAPI 应用
app = FastAPI(
    title="LLM信息抽取API",
    description="简单的领域识别、意图分类和槽位填充服务",
    version="1.0.0"
)

# 配置 OpenAI 客户端
client = openai.OpenAI(
    api_key="sk-4cb91fbab0154645a837fe4afb5d35ef",  # 替换为你的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义系统提示词
SYSTEM_PROMPT = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签

可用标签说明：
- 领域类别：用于分类文本所属的应用领域
- 意图类型：表示用户想要执行的操作类型
- 实体标签：文本中出现的具体信息片段

候选标签集合：
- 领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 意图类型：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

抽取规则：
1. 仔细分析文本内容，确定最合适的领域、意图和实体
2. 实体抽取要求：从文本中准确提取对应的实体值
3. 如果某个信息不存在，则不填写对应的字段
4. 严格遵循输出格式，只输出JSON

输出格式：
```json
{
    "domain": "领域标签",
    "intent": "意图标签", 
    "slots": {
        "实体标签1": "实体值1",
        "实体标签2": "实体值2"
    }
}

单文本示例：
输入："从许昌到中山的公交车"
输出：
json
{{
    "domain": "bus",
    "intent": "QUERY", 
    "slots": {{
        "startLoc_city": "许昌",
        "endLoc_city": "中山"
    }}
}}

多文本示例：
输入：1."请帮我打开uc",2."我想听昨天晚上的新闻。",3."最近有什么热门新闻了。"
输出：
[
  {
    "text": "请帮我打开uc",
    "domain": "app",
    "intent": "LAUNCH",
    "slots": {
      "name": "uc"
    }
  },
 {
    "text": "我想听昨天晚上的新闻。",
    "domain": "news",
    "intent": "PLAY",
    "slots": {
      "datetime_date": "昨天",
      "datetime_time": "晚上"
    }
  },
 {
    "text": "最近有什么热门新闻了。",
    "domain": "news",
    "intent": "PLAY",
    "slots": {
      "category": "热门",
      "datetime_date": "最近"
    }
  }]
现在请分析以下文本："""



#定义请求和响应模型
class PredictRequest(BaseModel):
  text: str

class PredictResponse(BaseModel):
  domain: str
  intent: str
  slots: Dict[str, str]
  text: str
  processing_time: float

def predict(text: str) -> Dict:
  """预测单个文本"""
  start_time = time.time()
  try:
      # 调用大模型API
      completion = client.chat.completions.create(
          model="qwen-plus",
          messages=[
              {"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": text},
          ],
          temperature=0.1,
      )

      result_content = completion.choices[0].message.content
      print(result_content )
      # 提取JSON部分
      json_match = re.search(r'```json\s*(.*?)\s*```', result_content, re.DOTALL)

      if json_match:
          json_str = json_match.group(1)
      else:
          json_str = result_content

      result = json.loads(json_str)
      processing_time = time.time() - start_time

      # 返回标准化结果
      return {
          "domain": result.get("domain", "unknown"),
          "intent": result.get("intent", "DEFAULT"),
          "slots": result.get("slots", {}),
          "text": text,
          "processing_time": round(processing_time, 2)
      }

  except Exception as e:
      # 如果出错，返回默认结果
      processing_time = time.time() - start_time
      return {
          "domain": "unknown",
          "intent": "DEFAULT",
          "slots": {},
          "text": text,
          "processing_time": round(processing_time, 2)
      }
print("单文本",predict('今天晚上会下雨'))
print("多文本",predict('1."糖醋鲤鱼怎么做啊？",2."请帮我打开百度",3."我想看昨天直播的LOL总决赛"'))
