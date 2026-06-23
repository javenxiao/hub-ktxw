import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import json
import ollama
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from fastmcp import Client
import fitz  # PyMuPDF，解析PDF文件提取文本


# 句向量模型路径
LOCAL_EMBED_MODEL_PATH = '/Users/wangyingyue/materials/大模型学习资料——八斗/models/shibing624-text2vec-base-chinese'
# rerank模型路径
# RERANK_MODEL_PATH = '/Users/wangyingyue/materials/大模型学习资料——八斗/models/Qwen/Qwen3-Reranker-8B'
RERANK_MODEL_PATH = '/Users/wangyingyue/materials/大模型学习资料——八斗/models/bge_models/BAAI/bge-reranker-base'
# 输出文件路径
RUN_LOG_JSONL_PATH = './run_log_final.jsonl'

TOP_K_RETRIEVAL = 3  # RAG检索返回最相关的3个文档
# 模型配置
LLM_MODEL_NAME = 'qwen3:0.6b'
# 设备
# device = torch.device("mps" if torch.backends.mps.is_available() else
#                       "cuda" if torch.cuda.is_available() else
#                       "cpu")
device = torch.device("cpu")

MCP_SERVER_URL = "http://localhost:8900/sse"

doc_files = [
    './documents/0a96883f-a7ee-4aa7-b020-efdb1f634d0.pdf',
    './documents/0b9a5088-426e-4c65-8bfb-ef98fc410da.pdf',
    './documents/1a316dd4-a53a-4f31-b13d-602d7a02581.pdf',
    './documents/1cc1229e-c0ea-45d0-b189-cef606f2413.pdf',
    './documents/1d4bb809-3292-4e24-9518-30a47c03565.pdf',
    './documents/4ab88c6c-d687-40ff-80c6-324a19d246a.pdf',
    './documents/5c148012-3024-49f5-b567-3614240697e.pdf',
    './documents/7afa380b-eb8d-4dde-b3bb-325e3091572.pdf',
    './documents/8adf20ab-1ffc-4b0f-818f-7ca7e5d22c9.pdf',
    './documents/9e24ccdd-8a28-41ec-95d1-4a25a603c49.pdf'
]

TOOLS = [
    "get_fish_population", "get_order", "get_crop_yield",
    "get_combustion_efficiency", "get_fuel_consumption", "get_satisfaction",
    "get_conversion_rate", "get_milk", "get_mastery_score", "get_average_travel_time"
]

TOOL_DESCRIPTIONS = {
    "get_fish_population": "用于水产养殖中预测鱼类种群数量，考虑内禀增长率、环境承载力和鱼苗投放量",
    "get_order": "用于电子商务中预测当日订单数量，基于广告支出、折扣力度和前一日订单数量",
    "get_crop_yield": "用于农业中预测作物产量，基于土壤肥力、灌溉量和气温",
    "get_combustion_efficiency": "用于能源化工中计算燃烧效率，基于燃料热值",
    "get_fuel_consumption": "用于汽车制造中预测百公里油耗，基于汽车重量",
    "get_satisfaction": "用于旅游领域评估游客满意度，基于天气、人流量、设施、安全性和风景",
    "get_conversion_rate": "用于金融领域计算复利，基于本金、利率和年限",
    "get_milk": "在模拟和预测奶牛在特定饲养条件下的日均产奶量，基于饲料质量、气温、健康评分、饮水时间",
    "get_mastery_score": "用于教育培训领域评估学习掌握度，基于学习时间",
    "get_average_travel_time": "用于交通运输领域预测通行时间，基于限速、天气因子、车辆密度和车道数量"
}

TOOL_PARAM_DESC = {
    "get_fish_population": "P_t（种群数量）、r（内禀增长率）、K（环境承载力）、S（鱼苗投放量）",
    "get_order": "ad_spend（广告支出）、discount_rate（折扣力度）、prev_orders（前一日订单数）",
    "get_crop_yield": "F（土壤肥力）、I（灌溉量）、T（气温）",
    "get_combustion_efficiency": "HV（燃料热值）",
    "get_fuel_consumption": "weight（汽车重量）",
    "get_satisfaction": "weather（天气）、crowd_level（人流量）、amenities（设施）、safety（安全）、scenery（风景）",
    "get_conversion_rate": "P（本金）、r（利率）、t（年限）",
    "get_milk": "quality（饲料质量）、T（气温）、R（健康评分）、t（饮水时间）",
    "get_mastery_score": "time（学习时间）",
    "get_average_travel_time": "v（限速）、w（天气因子）、p（车辆密度）、n（车道数）"
}


def build_simple_knowledge_base():
    knowledge_base = []
    for doc_path in tqdm(doc_files, desc='解析文档纯文本'):
        doc_id, full_text = os.path.basename(doc_path), ""
        try:
            with fitz.open(doc_path) as doc:
                for page in doc:
                    full_text += page.get_text() + "\n"

            knowledge_base.append({"id": doc_id, "full_text": full_text.strip()})
        except Exception:
            # 解析失败则文本为空，避免整个流程中断
            knowledge_base.append({"id": doc_id, "full_text": ""})
    return knowledge_base


def llm_match(question, top1_doc):
    tool_info = ""
    for tool_name in TOOLS:
        tool_desc = TOOL_DESCRIPTIONS[tool_name]
        tool_info += f"{tool_name}：{tool_desc}\n"

    prompt = f"""
    你是一个问题匹配专家，你的任务是：
    1. 仔细阅读用户提问和参考文档
    2. 参考工具介绍tool_info，从提供的工具列表TOOLS中选择**唯一**最匹配的工具
    3. **仅返回工具名称**，不要返回任何其他解释、标点或字符
    4. 如果没有匹配的工具，**仅返回"None"**
    工具列表为：{TOOLS}
    工具的介绍为：{tool_info}
    用户提问为：{question}
    
    参考文档为：{top1_doc['full_text']}
    """
    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0}
        )
        match_tool = response['message']['content'].strip()  # 提取 LLM 原始输出（去除首尾空格）

        return match_tool
    except Exception as e:
        print(f"大模型匹配tool失败：{e}")
        return None


def extract_params(question, matched_tool):
    prompt = f"""
    你是一个严格的参数提取器，必须**仅返回纯JSON格式**，不能包含任何其他文字、解释或标点。
    要求：
    1. 只返回JSON对象，不包含任何其他内容
    2. 参数名必须与工具定义的参数名完全一致，使用双引号包围
    3. 数值直接写，不需要引号
    4. 严格遵循JSON格式规范
    5. 确保所有必填参数都被提取
    
    工具信息：
    - 工具为：{matched_tool}
    - 工具的介绍为：{TOOL_DESCRIPTIONS[matched_tool]}
    - 工具的参数为：{TOOL_PARAM_DESC[matched_tool]}
    
    用户提问为：{question}
    """

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0}
        )
        params = json.loads(response['message']['content'].strip())
        for k, v in params.items():
            if v is not None:
                params[k] = float(v) if isinstance(v, (int, float)) else None

        return params
    except Exception as e:
        print(f"参数提取失败：{e}")
        return None

async def call_mcp_tool(tool_name: str, params: dict) -> float | None:
    """
    通过 MCP-SSE 协议调用远端工具
    """
    try:
        async with sse_client("http://localhost:8900/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize() 
                result = await session.call_tool(tool_name, arguments=params)
                if result.content and result.content[0].type == "text":
                    return float(result.content[0].text)
                return None
    except Exception as e:
        print(f"Tool 调用失败: {e}")
        return None


def main():
    knowledge_base = build_simple_knowledge_base()
    valid_knowledge_base = [doc for doc in knowledge_base if doc['full_text']]

    embed_model = SentenceTransformer(LOCAL_EMBED_MODEL_PATH)
    corpus_texts = [doc['full_text'] for doc in valid_knowledge_base]  # 提取所有有效文档的文本

    corpus_embeddings = torch.tensor(embed_model.encode(corpus_texts, show_progress_bar=True))

    question = '广告支出为100、当日折扣力度5%、前一日订单数量200，求当日订单数量的预测值'

    # 粗筛top3文档
    query_embedding = torch.tensor(embed_model.encode(question))
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_k_scores, top_k_indices = torch.topk(cos_scores, k=TOP_K_RETRIEVAL)
    top_k_docs = [valid_knowledge_base[i] for i in top_k_indices]

    # rerank精筛与提问最相似文档
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_PATH).to(device)
    # rerank_model = AutoModelForSequenceClassification.from_pretrained(
    #     RERANK_MODEL_PATH,
    #     torch_dtype=torch.float16
    # ).to(device)
    rerank_model.eval()

    pairs = []
    for doc in top_k_docs:
        doc_text = doc['full_text']
        pairs.append([question, doc_text])

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=256
    )

    with torch.no_grad():
        rerank_scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    for i, doc in enumerate(top_k_docs):
        doc['rerank_score'] = rerank_scores[i].item()

    top1_doc = sorted(top_k_docs, key=lambda x: x['rerank_score'], reverse=True)[0]

    print(f"最相关文档：{top1_doc['id']}\n")
    print(f"最相关文档内容：\n{top1_doc['full_text'][: 200]}...\n")

    # 大模型匹配tool
    match_tool = llm_match(question, top1_doc)
    if not match_tool:
        raise ValueError("未匹配到tool")
    print(f"匹配到的工具：{match_tool}\n")

    # 大模型提取参数
    params = extract_params(question, match_tool)
    if not params:
        raise ValueError("无法提取有效参数")
    print(f"提取出的参数为：{params}\n")

    # 调用MCP Tool
    result = asyncio.run(call_mcp_tool(match_tool, params))
    if result is None:
        raise ValueError("Tool调用未得到正确结果")

    print(f"用户问题为：{question}\n")
    print(f"调用工具为：{match_tool}\n")
    print(f"计算结果为：{result}\n")


if __name__ == "__main__":
    main()
