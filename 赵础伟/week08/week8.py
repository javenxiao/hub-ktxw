from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from coze import CozeClient

# 初始化 Coze 客户端，指定使用 Qwen 模型
client = CozeClient(api_key='sk-41357d21198744d38f05c076348fab3a', model="qwen")

# FastAPI 应用实例
app = FastAPI()

# 提示词设计
intent_prompt = """
你是一个智能助手，用户会向你提出问题。你的任务是识别用户的意图。
问题: {user_input}
请回答用户的意图，例如: '查询天气', '预订酒店', '购买商品'。
"""

domain_prompt = """
你是一个智能助手，用户会向你提出问题。你的任务是识别问题所属的领域。
问题: {user_input}
请回答问题的领域，例如: '旅游', '电商', '教育'。
"""

entity_prompt = """
你是一个智能助手，用户会向你提出问题。你的任务是从问题中提取关键实体。
问题: {user_input}
请提取问题中的实体，例如: '地点', '时间', '商品名称'。
"""

# 定义任务函数
def identify_intent(user_input):
    prompt = intent_prompt.format(user_input=user_input)
    response = client.generate(prompt=prompt)
    return response['text']

def identify_domain(user_input):
    prompt = domain_prompt.format(user_input=user_input)
    response = client.generate(prompt=prompt)
    return response['text']

def identify_entities(user_input):
    prompt = entity_prompt.format(user_input=user_input)
    response = client.generate(prompt=prompt)
    return response['text']

# 定义请求模型
class UserInput(BaseModel):
    input: str

# FastAPI 路由
@app.post("/identify")
async def identify(user_input: UserInput):
    try:
        # 获取用户输入
        input_text = user_input.input

        # 执行任务
        intent = identify_intent(input_text)
        domain = identify_domain(input_text)
        entities = identify_entities(input_text)

        # 返回结果
        return {
            "input": input_text,
            "intent": intent.strip(),
            "domain": domain.strip(),
            "entities": entities.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 测试路由
@app.get("/")
async def root():
    return {"message": "意图识别服务已启动"}