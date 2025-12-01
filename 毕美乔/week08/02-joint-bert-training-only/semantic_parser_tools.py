import json
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, create_model
from typing_extensions import Literal
from openai import OpenAI

# --------------------------
# 初始化客户端
# --------------------------
OPENAI_API_KEY = "sk-******"
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --------------------------
# ExtractionAgent
# --------------------------
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        # 生成工具 schema
        model_schema = response_model.model_json_schema()
        properties = model_schema['properties']
        required_fields = model_schema.get('required', list(properties.keys()))

        tools = [
            {
                "type": "function",
                "function": {
                    "name": model_schema["title"],
                    "description": model_schema.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_fields
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="auto",
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print("ERROR:", e)
            print("LLM 原始输出:", response.choices[0].message)
            return None

# ----------------- 通用 Domain/Intent 模型 -----------------
class DomainIntentModel(BaseModel):
    domain: Literal["music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website", "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke", "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus", "story"] = Field(description="领域")
    intent: Literal["OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"] = Field(description="意图")


# ----------------- 动态生成 Slots 模型 -----------------
def generate_slots_model(domain: str, domain_slot_map: Dict[str, List[str]]):
    fields = {}
    for slot in domain_slot_map.get(domain, []):
        fields[slot] = (Optional[str], Field(description=f"{slot}槽位"))
    model = create_model(f"{domain}_SlotsModel", **fields)
    return model

# --------------------------
# 工具：读取文件
# --------------------------
def _load_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"⚠️ 文件未找到: {file_path}")
        return []

def _load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --------------------------
# 加载配置
# --------------------------
domain_file = 'domains.txt'
intent_file = 'intents.txt'
domain_slot_map_file = 'domain_slot_map.json'

domains = _load_list(domain_file)
intents = _load_list(intent_file)
domain_slot_map = _load_json(domain_slot_map_file)

# --------------------------
# 示例调用
# --------------------------
if __name__ == "__main__":
    text = "帮我查询下从北京到天津到武汉的汽车票"
    text = '来一首钟汉良的视觉动物'


    # 阶段一：抽取 domain + intent
    agent = ExtractionAgent(model_name="qwen-plus")
    domain_intent_result = agent.call(text, DomainIntentModel)

    if domain_intent_result:
        current_domain = domain_intent_result.domain
        intent = domain_intent_result.intent
        # 阶段二：根据 domain 动态生成 Slots 模型
        SlotsModel = generate_slots_model(current_domain, domain_slot_map)

        slots_result = agent.call(text, SlotsModel)
        result_dict = slots_result.model_dump()  # 将 BaseModel 转成 dict
        filtered_slots = {k: v for k, v in result_dict.items() if v not in (None, "", [])}

        result = {
            "intent": intent,
            "domain": current_domain,
            "slots": filtered_slots
        }
        print("\n🧠 输入：", text)
        print("📦 输出：")
        print(json.dumps(result, ensure_ascii=False, indent=2))

