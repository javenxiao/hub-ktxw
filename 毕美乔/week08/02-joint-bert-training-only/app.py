from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from semantic_parser_prompt import SemanticParser
from semantic_parser_tools import ExtractionAgent, DomainIntentModel, generate_slots_model, domain_slot_map
from router_schemas import QueryRequest, QueryResponse
import os

app = FastAPI(
    title="Semantic Parsing Service",
    description="支持 prompt 模式与 function-calling 模式的语义解析服务",
    version="1.0.0"
)

# export OPENAI_API_KEY="sk-******"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("请在环境变量中设置 OPENAI_API_KEY")


@app.post("/parse_by_prompt")
def parse_text_by_prompt(query: QueryRequest):
    """
    通过 prompt 实现的语义解析方法
    """
    parser = SemanticParser(api_key=OPENAI_API_KEY)
    return parser.parse(query.text)


@app.post("/parse_by_schema")
def parse_text_by_schema(request: QueryRequest):
    """
    使用 function calling + 动态 Pydantic schema 的解析方法
    """
    agent = ExtractionAgent(model_name="qwen-plus")
    domain_intent_result = agent.call(request.text, DomainIntentModel)

    if not domain_intent_result:
        raise HTTPException(status_code=500, detail="无法识别 domain/intent")
    current_domain = domain_intent_result.domain
    intent = domain_intent_result.intent
    # 阶段二：根据 domain 动态生成 Slots 模型
    SlotsModel = generate_slots_model(current_domain, domain_slot_map)

    slots_result = agent.call(request.text, SlotsModel)
    if not slots_result:
        filtered_slots = {}
    else:
        result_dict = slots_result.model_dump()  # 将 BaseModel 转成 dict
        filtered_slots = {k: v for k, v in result_dict.items() if v not in (None, "", [])}

    result = {
        "text": request.text,
        "intent": intent,
        "domain": current_domain,
        "slots": filtered_slots
    }
    return result
