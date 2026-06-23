import asyncio
import os
os.environ["OPENAI_API_KEY"] = "sk-b8f16de2371547c397349552ecffce68"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import json
import requests
import urllib.parse
from typing import List, Dict, Any

# 假设以下导入能够正常工作，它们通常来自 agents 库
from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
    set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = "qwen-max"  # 假设这是 AliCloud 兼容模式下的一个模型名称
API_KEY = os.getenv("OPENAI_API_KEY", "sk-b8f16de2371547c397349552ecffce68")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化 AsyncOpenAI 客户端
llm_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 定义模型设置
model_settings = ModelSettings(
    model=MODEL_NAME,
    client=llm_client,
    temperature=0.3
)

# --- 2. 外部工具（Jina Search & Crawl） ---
JINA_API_KEY = "jina_8918effb420d4bff8530c9d9f3bbe536NWhiCZdKQFNgoFLd4aganV1XnsaA"

def search_jina(query: str) -> str:
    """通过jina进行谷歌搜索，返回JSON格式的搜索结果字符串"""
    print(f"-> [Jina Search] 正在搜索: {query[:50]}...")
    try:
        # 确保查询参数是 URL 编码的
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/?q={encoded_query}&hl=zh-cn"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "no-content"  # Jina Search 默认返回摘要和引用
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # 抛出 HTTP 错误

        # Jina Search 返回的是一个包含结果的 JSON 结构，提取关键信息
        results = response.json().get('data', [])

        # 提取标题、链接和摘要
        formatted_results = []
        for res in results:
            formatted_results.append({
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "snippet": res.get("content", "")
            })

        return json.dumps(formatted_results, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        print(f"Error during Jina Search: {e}")
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)
    except Exception as e:
        print(f"Unexpected error in Jina Search: {e}")
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)


def crawl_jina(url: str) -> str:
    """通过jina抓取完整网页内容，返回Markdown格式的文本"""
    print(f"-> [Jina Crawl] 正在抓取: {url[:50]}...")
    try:
        # Jina Reader API
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "content",  # 请求返回完整内容
            "X-Content-Type": "markdown"  # 请求返回 Markdown 格式
        }
        # 使用 r.jina.ai 作为代理
        response = requests.get("https://r.jina.ai/" + url, headers=headers, timeout=20)
        response.raise_for_status()

        # 返回内容通常在 'data' 字段的 'content' 中
        content = response.json().get("data", {}).get("content", f"无法抓取 URL: {url} 的内容。")

        return content
    except requests.exceptions.RequestException as e:
        print(f"Error during Jina Crawl for {url}: {e}")
        return f"抓取失败: {e}"
    except Exception as e:
        print(f"Unexpected error in Jina Crawl for {url}: {e}")
        return f"抓取失败: {e}"


# 将同步函数包装成异步，以便在 Agents 异步环境中使用
async def async_search_jina(query: str) -> str:
    """异步调用 Jina 搜索"""
    # 使用asyncio.to_thread将同步操作放入线程池执行，避免阻塞事件循环
    return await asyncio.to_thread(search_jina, query)


async def async_crawl_jina(url: str) -> str:
    """异步调用 Jina 抓取"""
    return await asyncio.to_thread(crawl_jina, url)

external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --- 3. 代理定义 (Agents) ---
orchestrator_system_prompt = """
你是一名深度研究专家和项目经理。你的任务是协调整个研究项目，包括：
1. **研究规划 (生成大纲):** 根据用户提供的研究主题和初步搜索结果，生成一个详尽、逻辑严密、结构清晰的报告大纲。大纲必须以严格的 JSON 格式输出，用于指导后续的章节内容检索和起草工作。
2. **报告整合 (组装):** 在所有章节内容起草完成后，将它们整合在一起，形成一篇流畅、连贯、格式优美的最终研究报告。报告必须包括摘要、完整的章节内容、结论和引用来源列表。
"""
DeepResearchAgent = Agent(
    "Deep Research Orchestrator",
    instructions=orchestrator_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

# 3.2. 内容起草代理 (Drafting Agent)
drafting_system_prompt = """
你是一名专业的内容撰稿人。你的任务是将提供的原始网页抓取内容和搜索结果，根据指定的章节主题，撰写成一篇结构合理、重点突出、信息准确的报告章节。
你必须严格遵守以下规则：
1. **聚焦主题:** 严格围绕给定的 '章节主题' 进行撰写。
2. **信息来源:** 只能使用提供的 '原始网页内容' 和 '搜索结果摘要' 中的信息。
3. **格式:** 使用 Markdown 格式。
4. **引用:** 对于文中引用的关键事实和数据，必须在段落末尾用脚注或括号标记引用的来源 URL，例如 [来源: URL]。
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

# 3.3. 内容反馈代理
feedback_system_prompt = """
你是一名专业的内容反馈专家。你的任务是对提供的每个章节内容进行具体的评价及改进建议。
你必须严格遵守以下规则：
1. **反馈：** 根据内容从以下个维度进行合理且客观的评价：
    - 是否基于提供的材料
    - 是否严格围绕‘章节主题’
    - 对于中文引用的关键事实和数据，是否在段落末尾用脚注或括号标记引用的来源 URL
    - 语言是否流畅、符合逻辑
2. **信息来源：** 只能对提供的‘章节内容’进行评价和反馈。
3. **格式：** 使用 markdown 格式。
"""
FeedbackAgent = Agent(
    "Content Feedback Specialist",
    instructions=feedback_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    )
)


# 章节处理
async def handle_section(section: Dict[str, str]) -> str:
    section_title = section.get("section_title")
    search_keywords = section.get("search_keywords")
    print(f"\n--- 开始并发处理章节: {section_title} ---")

    try:
        # 3.1. 精确检索
        section_query = f"{section_title} 搜索关键词: {search_keywords}"
        section_search_results_str = await async_search_jina(section_query)  # 根据之前 Agent 总结好的章节主题和关键词再去进行搜索

        # 3.2. 筛选并抓取前2个链接
        try:
            # 解析搜索结果 JSON，提取 URL 列表
            search_results = json.loads(section_search_results_str)
            # 只保留前两个URL
            urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
        except:
            print("Warning: Failed to parse search results for crawl.")
            urls_to_crawl = []

        crawled_content = []
        for url in urls_to_crawl:
            content = await async_crawl_jina(url)
            # 限制内容长度为 3000 字符，避免模型输入过长
            crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")  # 限制抓取内容长度

        raw_materials = "\n\n".join(crawled_content)

        # 3.3. 内容起草 (调用 Drafting Agent)
        draft_prompt = f"""
                **章节主题:** {section_title}
    
                **搜索结果摘要:**
                {section_search_results_str[:3000]}... (仅用于参考要点)
    
                **原始网页内容 (请基于此内容撰写):**
                {raw_materials}
    
                请根据上述信息，撰写 {section_title} 这一章节的详细内容。
                """


        section_draft = await Runner.run(
            DraftingAgent,
            draft_prompt,
        )
        current_draft = section_draft.final_output
        print(f"-> 章节起草完成: {section_title}")
        print(section_draft.final_output)
        print("")

        # 3.4. 内容反馈
        try:
            section_feedback = await Runner.run(
                FeedbackAgent,
                f"""**章节内容（请基于此内容进行反馈）: ** {current_draft}"""
            )
            print(f"-> 获取反馈完成:\n {section_feedback.final_output}")
            print("")
        except Exception as e:
            error_msg = f"章节反馈失败: {e}"
            print(f"-> {error_msg}")

        return f"## {section_title}\n\n{current_draft}"

    except Exception as e:
        error_msg = f"章节处理失败: {e}"
        return f"## {section_title}\n\n{error_msg}"


# --- 4. 深度研究核心流程 ---

async def deep_research(query: str, max_sections: int = 5) -> str:
    """
    执行深度研究流程：规划 -> 检索 -> 抓取 -> 起草 -> 整合。
    """
    print(f"\n--- Deep Research for: {query} ---\n")

    # 1. 初步检索
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)
    print(initial_search_results_str)

    # 2. 生成研究大纲 (使用 JSON 模式确保结构化输出)
    print("\nStep 2: 基于初步结果生成研究大纲...")

    init_prompt = f"""研究主题: {query}
初步搜索结果摘要: {initial_search_results_str}
"""

    outline_prompt = init_prompt + """请根据上述信息，生成一个详细的报告大纲。大纲必须包含一个 'title' 和一个 'sections' 数组。
每个章节对象必须包含 'section_title' 和 'search_keywords' (用于精确检索的关键词)。

示例输出 JSON 格式如下，只要json，不要有其他输出
{
    "title": "关于 XX 的深度研究报告",
    "sections": [
        {"section_title": "引言与背景", "search_keywords": "历史, 现状"},
        {"section_title": "核心要素与机制", "search_keywords": "关键概念, 工作原理"},
        {"section_title": "应用与影响", "search_keywords": "行业应用, 社会影响"}
    ]
}
"""
    try:
        # 调用 Orchestrator Agent 生成 JSON 格式的大纲
        outline_response = await Runner.run(
            DeepResearchAgent,
            outline_prompt,
        )
        print(outline_response)
        # 使用 json.loads() 将 JSON 字符串转换为 Python 字典
        # 去除可能包含的 Markdown 代码块标记（```json 和 ```）
        outline_json = json.loads(outline_response.final_output.strip("```json").strip("```"))

    except Exception as e:
        print(f"Error generating outline: {e}. Falling back to a simple structure.")
        # 失败时提供默认大纲
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
    sections = outline_json.get("sections", [])
    # 如果章节数超过 max_sections，只保留前 max_sections 个章节
    if len(sections) > max_sections:
        sections = sections[:max_sections]

    print(f"报告标题: {research_title}")
    print(f"规划了 {len(sections)} 个章节。")

    print("\nStep 3: 并发处理所有章节...")
    tasks = [handle_section(section) for section in sections]
    drafted_sections = await asyncio.gather(*tasks)

    # 4. 报告整合与最终输出 (调用 Orchestrator Agent)
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)

    final_prompt = f"""
    请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。

    **报告标题:** {research_title}

    **已起草的章节内容:**
    {full_report_draft}

    **任务要求:**
    1. 在报告开头添加一个**【摘要】**，总结报告的主要发现和结论。
    2. 保持各章节之间的连贯性。
    3. 在报告末尾添加一个**【结论与展望】**部分（如果大纲中没有）。
    4. 添加一个**【引用来源】**列表，列出所有章节中提到的 URL。
    5. 整体报告必须格式优美，使用 Markdown 格式。
    """

    try:
        final_report = await Runner.run(
            DeepResearchAgent,
            final_prompt,
        )
        return final_report.final_output
    except Exception as e:
        return f"最终报告整合失败: {e}\n\n已完成的章节草稿:\n{full_report_draft}"


async def main():
    research_topic = "Agentic AI在软件开发中的最新应用和挑战"
    final_report = await deep_research(research_topic)
    print(final_report)


# 使用 Runner 启动异步主函数
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except NameError:
        # Fallback to standard asyncio run if Runner is not defined or preferred
        asyncio.run(main())
