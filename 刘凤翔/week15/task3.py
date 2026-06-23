import asyncio
import os
import json
import requests
import urllib.parse
from typing import List, Dict, Any
from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
    set_default_openai_api, set_tracing_disabled

# 环境配置
os.environ["OPENAI_API_KEY"] = "sk-c4395731abd4446b8642c7734c8dbf56"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = "qwen-max"
API_KEY = os.getenv("OPENAI_API_KEY", "sk-c4395731abd4446b8642c7734c8dbf56")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化客户端
llm_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
external_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL)

# 模型设置
model_settings = ModelSettings(model=MODEL_NAME, client=llm_client, temperature=0.3)

# Jina 搜索和抓取工具
JINA_API_KEY = "jina_8918effb420d4bff8530c9d9f3bbe536NWhiCZdKQFNgoFLd4aganV1XnsaA"


def search_jina(query: str) -> str:
    """通过jina进行谷歌搜索，返回JSON格式的搜索结果字符串"""
    print(f"-> [Jina Search] 正在搜索: {query[:50]}...")
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/?q={encoded_query}&hl=zh-cn"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "no-content"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        results = response.json().get('data', [])
        formatted_results = [{
            "title": res.get("title", ""),
            "url": res.get("url", ""),
            "snippet": res.get("content", "")
        } for res in results]

        return json.dumps(formatted_results, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)


def crawl_jina(url: str) -> str:
    """通过jina抓取完整网页内容，返回Markdown格式的文本"""
    print(f"-> [Jina Crawl] 正在抓取: {url[:50]}...")
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "content",
            "X-Content-Type": "markdown"
        }
        response = requests.get("https://r.jina.ai/" + url, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json().get("data", {}).get("content", f"无法抓取 URL: {url} 的内容。")
    except Exception as e:
        return f"抓取失败: {e}"


async def async_search_jina(query: str) -> str:
    return await asyncio.to_thread(search_jina, query)


async def async_crawl_jina(url: str) -> str:
    return await asyncio.to_thread(crawl_jina, url)


# 代理定义
orchestrator_system_prompt = """
你是一名深度研究专家和项目经理。你的任务是协调整个研究项目，包括：
1. **研究规划 (生成大纲):** 根据用户提供的研究主题和初步搜索结果，生成详尽的报告大纲。
2. **报告整合 (组装):** 在所有章节内容起草完成后，将它们整合在一起，形成完整报告。
"""
DeepResearchAgent = Agent(
    "Deep Research Orchestrator",
    instructions=orchestrator_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client)
)

drafting_system_prompt = """
你是一名专业的内容撰稿人。你的任务是根据提供的原始网页内容和搜索结果，撰写指定章节。
严格遵守：聚焦主题、基于提供信息、使用Markdown格式、标注引用来源。
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client)
)

feedback_system_prompt = """
你是一名内容质量审核专家。请评估章节内容是否符合要求，并提供具体改进建议。
评估维度：1.主题相关性 2.信息完整性 3.结构合理性 4.引用规范性
输出格式：先判断是否需要修改（是/否），然后提供具体改进建议。
"""
FeedbackAgent = Agent(
    "Content Feedback Specialist",
    instructions=feedback_system_prompt,
    model=OpenAIChatCompletionsModel(model="qwen-max", openai_client=external_client)
)


async def process_section(section: Dict[str, str], max_iterations: int = 3) -> str:
    """处理单个章节，包含多轮反馈优化"""
    section_title = section["section_title"]
    search_keywords = section["search_keywords"]
    print(f"\n--- 开始处理章节: {section_title} ---")

    # 获取章节相关资料
    section_query = f"{section_title} 搜索关键词: {search_keywords}"
    search_results_str = await async_search_jina(section_query)

    try:
        search_results = json.loads(search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
    except:
        urls_to_crawl = []
        print("警告: 无法解析搜索结果")

    # 抓取网页内容
    crawled_content = []
    for url in urls_to_crawl:
        content = await async_crawl_jina(url)
        crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")
    raw_materials = "\n\n".join(crawled_content)

    # 多轮迭代优化
    current_draft = ""
    for iteration in range(max_iterations):
        print(f"-> 章节 {section_title} 第 {iteration + 1} 轮生成")

        # 生成或优化草稿
        draft_prompt = f"""
        **章节主题:** {section_title}
        **搜索结果摘要:** {search_results_str[:3000]}...
        **原始网页内容:** {raw_materials}
        {f'**上一轮反馈:** {feedback}' if iteration > 0 else ''}
        {f'**当前草稿:** {current_draft}' if iteration > 0 else ''}
        请根据上述信息，{'优化' if iteration > 0 else '撰写'} {section_title} 章节内容。
        """

        draft_response = await Runner.run(DraftingAgent, draft_prompt)
        current_draft = draft_response.final_output

        # 获取反馈
        feedback_prompt = f"""
        **章节主题:** {section_title}
        **章节内容:** {current_draft}
        请评估以上章节内容是否符合要求，并提供具体改进建议。
        """
        feedback_response = await Runner.run(FeedbackAgent, feedback_prompt)
        feedback = feedback_response.final_output
        print(f"-> 章节 {section_title} 反馈: {feedback[:100]}...")

        # 检查是否需要继续优化
        if "否" in feedback.split("\n")[0]:
            print(f"-> 章节 {section_title} 已满足要求，停止迭代")
            break

    return f"## {section_title}\n\n{current_draft}"


async def deep_research(query: str, max_sections: int = 5, max_iterations: int = 3) -> str:
    """改进后的深度研究流程：并行处理章节，加入反馈迭代机制"""
    print(f"\n--- 开始深度研究: {query} ---\n")

    # 1. 初步检索
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)

    # 2. 生成研究大纲
    print("\nStep 2: 生成研究大纲...")
    init_prompt = f"研究主题: {query}\n初步搜索摘要: {initial_search_results_str}"
    outline_prompt = init_prompt + """请生成包含'title'和'sections'的JSON格式报告大纲，
每个章节包含'section_title'和'search_keywords'。示例格式：
{
    "title": "关于XX的研究报告",
    "sections": [{"section_title": "引言", "search_keywords": "关键词"}]
}
"""
    try:
        outline_response = await Runner.run(DeepResearchAgent, outline_prompt)
        outline_json = json.loads(outline_response.final_output.strip("```json").strip("```"))
    except Exception as e:
        print(f"生成大纲出错: {e}，使用默认大纲")
        outline_json = {
            "title": f"关于 {query} 的深度研究报告",
            "sections": [
                {"section_title": "引言与背景", "search_keywords": f"{query}, 历史, 现状"},
                {"section_title": "核心要素与机制", "search_keywords": f"{query}, 工作原理, 关键技术"},
                {"section_title": "应用与影响", "search_keywords": f"{query}, 行业应用, 社会影响"},
                {"section_title": "结论与展望", "search_keywords": f"{query}, 发展趋势, 挑战"}
            ]
        }

    research_title = outline_json.get("title", f"关于 {query} 的深度研究报告")
    sections = outline_json.get("sections", [])[:max_sections]
    print(f"报告标题: {research_title}")
    print(f"规划章节数: {len(sections)}")

    # 3. 并行处理所有章节（带反馈迭代）
    print("\nStep 3: 并行处理所有章节...")
    section_tasks = [process_section(section, max_iterations) for section in sections]
    drafted_sections = await asyncio.gather(*section_tasks)

    # 4. 整合最终报告
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)
    final_prompt = f"""
    请整合以下章节为完整报告：
    **报告标题:** {research_title}
    **章节内容:** {full_report_draft}
    要求：1. 添加【摘要】 2. 保持连贯性 3. 添加【结论与展望】（如无） 4. 添加【引用来源】 5. 使用Markdown格式
    """

    try:
        final_report = await Runner.run(DeepResearchAgent, final_prompt)
        return final_report.final_output
    except Exception as e:
        return f"整合报告失败: {e}\n\n章节草稿:\n{full_report_draft}"


async def main():
    research_topic = "Agentic AI在软件开发中的最新应用和挑战"
    final_report = await deep_research(research_topic, max_sections=5, max_iterations=3)
    print(final_report)


if __name__ == "__main__":
    asyncio.run(main())