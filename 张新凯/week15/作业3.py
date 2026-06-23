import asyncio
import os

os.environ["OPENAI_API_KEY"] = "sk-c4395731abd4446b8642c7734c8dbf56"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import json
import requests
import urllib.parse
from typing import Dict, Any

# 假设以下导入能够正常工作，它们通常来自 agents 库
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
    set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = "qwen-max"  # 假设这是 AliCloud 兼容模式下的一个模型名称
API_KEY = os.getenv("OPENAI_API_KEY", "sk-c4395731abd4446b8642c7734c8dbf56")
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
5. **结构:** 章节应包含清晰的子标题，逻辑连贯，信息完整。
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

# 3.3. 质量评估代理 (Quality Evaluation Agent)
evaluation_system_prompt = """
你是一名资深研究质量评估专家。你的任务是对生成的章节内容进行专业评估，并提供具体的改进建议。
请从以下维度进行评估：
1. **内容质量:** 信息是否准确、全面？是否包含了该主题的关键要点？
2. **结构逻辑:** 章节结构是否清晰？逻辑是否连贯？是否包含适当的子标题？
3. **信息来源:** 是否充分引用了原始资料？引用格式是否正确？
4. **深度与广度:** 分析是否深入？是否覆盖了该主题的主要方面？
5. **可读性:** 语言表达是否清晰流畅？技术术语解释是否恰当？

请按以下JSON格式提供评估结果和建议：
{
    "score": 0-100的评分,
    "strengths": ["优点1", "优点2", ...],
    "weaknesses": ["不足1", "不足2", ...],
    "specific_suggestions": ["具体改进建议1", "具体改进建议2", ...],
    "needs_improvement": true/false (是否需要重新生成或大幅修改)
}
"""
EvaluationAgent = Agent(
    "Quality Evaluation Specialist",
    instructions=evaluation_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)


# --- 4. 深度研究核心流程 ---

async def process_section(section: Dict[str, Any], iteration: int = 1, max_iterations: int = 3,
                          improvement_prompt: str = None, raw_materials: str = None) -> Dict[str, Any]:
    """
    处理单个章节的完整流程：检索 -> 抓取 -> 起草 -> 评估 -> (如果需要)重新生成
    """
    section_title = section.get("section_title")
    search_keywords = section.get("search_keywords")

    print(f"\n--- 处理章节: {section_title} (迭代 {iteration}) ---")

    if improvement_prompt:
        draft_prompt = improvement_prompt
    else:
        # 1. 检索并抓取章节相关材料
        raw_materials, section_search_results_str = await search_materials(search_keywords, section_title)

        # 2. 内容起草
        draft_prompt = f"""
        **章节主题:** {section_title}
    
        **搜索结果摘要:**
        {section_search_results_str[:3000]}... (仅用于参考要点)
    
        **原始网页内容 (请基于此内容撰写):**
        {raw_materials}
    
        请根据上述信息，撰写 {section_title} 这一章节的详细内容。
        """

    try:
        section_draft = await Runner.run(
            DraftingAgent,
            draft_prompt,
        )
        draft_content = section_draft.final_output
        print(f"-> 章节起草完成: {section_title}")
        print(section_draft.final_output)
        print("")

        # 3. 质量评估
        evaluation_prompt = f"""
        **章节主题:** {section_title}

        **章节内容:**
        {draft_content}

        **原始材料摘要:**
        {raw_materials[:2000]}...

        请对该章节内容进行全面评估，并提供具体的改进建议。
        """

        evaluation_result = await Runner.run(
            EvaluationAgent,
            evaluation_prompt,
        )

        # 解析评估结果
        try:
            # 提取JSON部分
            eval_text = evaluation_result.final_output.strip("```json").strip("```")
            eval_data = json.loads(eval_text)

            print(f"评估结果 - 分数: {eval_data.get('score', 0)}")
            print(f"优点: {', '.join(eval_data.get('strengths', []))}")
            print(f"改进建议: {', '.join(eval_data.get('specific_suggestions', []))}")

            needs_improvement = eval_data.get('needs_improvement', False)
            score = eval_data.get('score', 0)

            # 判断是否需要重新生成
            if iteration < max_iterations and (needs_improvement or score < 70):
                print(f"章节 {section_title} 需要改进，将进行重新生成 (迭代 {iteration + 1})")

                # 准备改进建议
                suggestions = eval_data.get('specific_suggestions', [])
                if suggestions:
                    improvement_prompt = f"""
                    **章节主题:** {section_title}

                    **之前的草稿:**
                    {draft_content}

                    **评估反馈与改进建议:**
                    {chr(10).join(suggestions)}

                    **原始材料:**
                    {raw_materials[:3000]}...

                    **任务:**
                    请根据上述评估反馈和建议，重新撰写这个章节。特别注意改进以下方面：
                    {chr(10).join(suggestions[:3])}
                    """

                    # 递归调用，进行重新生成
                    return await process_section(section, iteration + 1, max_iterations, improvement_prompt,
                                                 raw_materials)

        except json.JSONDecodeError:
            print("Warning: Failed to parse evaluation result as JSON")
            eval_data = {
                "score": 60,
                "needs_improvement": iteration < max_iterations,
                "specific_suggestions": ["评估结果格式错误，请重新生成"]
            }

            if iteration < max_iterations:
                return await process_section(section, iteration + 1, max_iterations)

        return {
            "section_title": section_title,
            "content": f"## {section_title}\n\n{draft_content}",
            "evaluation": eval_data,
            "iteration": iteration,
            "raw_materials": raw_materials[:1000]  # 保存部分原始材料用于最终整合
        }

    except Exception as e:
        error_msg = f"章节处理失败: {e}"
        print(f"-> {error_msg}")
        return {
            "section_title": section_title,
            "content": f"## {section_title}\n\n{error_msg}",
            "evaluation": {"score": 0, "needs_improvement": False},
            "iteration": iteration,
            "raw_materials": ""
        }


async def search_materials(search_keywords, section_title):
    # 1. 精确检索
    section_query = f"{section_title} 搜索关键词: {search_keywords}"
    section_search_results_str = await async_search_jina(section_query)
    # 2. 筛选并抓取前2个链接
    try:
        search_results = json.loads(section_search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
    except:
        print("Warning: Failed to parse search results for crawl.")
        urls_to_crawl = []
    # 并发抓取多个URL
    crawl_tasks = [async_crawl_jina(url) for url in urls_to_crawl]
    crawled_contents = await asyncio.gather(*crawl_tasks)
    raw_materials = "\n\n".join([f"--- URL: {url} ---\n{content[:3000]}..."
                                 for url, content in zip(urls_to_crawl, crawled_contents)])
    return raw_materials, section_search_results_str


async def deep_research(query: str, max_sections: int = 5) -> str:
    """
    执行深度研究流程：规划 -> 并发处理各章节 -> 整合。
    """
    print(f"\n--- Deep Research for: {query} ---\n")

    # 1. 初步检索
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)
    print(initial_search_results_str)

    # 2. 生成研究大纲
    print("\nStep 2: 基于初步结果生成研究大纲...")

    outline_prompt = f"""研究主题: {query}
初步搜索结果摘要: {initial_search_results_str[:2000]}...

请根据上述信息，生成一个详细的报告大纲。大纲必须包含一个 'title' 和一个 'sections' 数组。
每个章节对象必须包含 'section_title' 和 'search_keywords' (用于精确检索的关键词)。

示例输出 JSON 格式如下：
{{
    "title": "关于 XX 的深度研究报告",
    "sections": [
        {{"section_title": "引言与背景", "search_keywords": "历史, 现状"}},
        {{"section_title": "核心要素与机制", "search_keywords": "关键概念, 工作原理"}},
        {{"section_title": "应用与影响", "search_keywords": "行业应用, 社会影响"}}
    ]
}}

请只输出JSON，不要有其他内容。
"""
    try:
        # 调用 Orchestrator Agent 生成 JSON 格式的大纲
        outline_response = await Runner.run(
            DeepResearchAgent,
            outline_prompt,
        )
        print(outline_response)
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
    if len(sections) > max_sections:
        sections = sections[:max_sections]

    print(f"报告标题: {research_title}")
    print(f"规划了 {len(sections)} 个章节。")

    # 3. 并发处理所有章节
    print("\nStep 3: 并发处理所有章节...")

    # 创建所有章节的处理任务
    section_tasks = [process_section(section) for section in sections]

    # 并发执行所有任务
    section_results = await asyncio.gather(*section_tasks)

    # 4. 输出章节处理结果
    print("\n--- 章节处理完成 ---")
    drafted_sections = []
    all_evaluations = []

    for i, result in enumerate(section_results):
        section_title = result.get("section_title", f"章节 {i + 1}")
        evaluation = result.get("evaluation", {})
        iteration = result.get("iteration", 1)

        print(f"\n章节 {i + 1}: {section_title}")
        print(f"  迭代次数: {iteration}")
        print(f"  最终评分: {evaluation.get('score', 'N/A')}")
        print(f"  优点: {', '.join(evaluation.get('strengths', [])[:2])}")

        drafted_sections.append(result.get("content", ""))
        all_evaluations.append({
            "section_title": section_title,
            "evaluation": evaluation,
            "iteration": iteration
        })

    # 5. 报告整合与最终输出
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)

    # 准备评估总结
    evaluation_summary = "章节质量评估总结:\n"
    for eval_info in all_evaluations:
        evaluation_summary += f"- {eval_info['section_title']}: 评分{eval_info['evaluation'].get('score', 'N/A')}, 迭代{eval_info['iteration']}次\n"

    final_prompt = f"""
    请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。

    **报告标题:** {research_title}

    **章节质量评估总结:**
    {evaluation_summary}

    **已起草的章节内容:**
    {full_report_draft}

    **任务要求:**
    1. 在报告开头添加一个**【摘要】**，总结报告的主要发现和结论。
    2. 保持各章节之间的连贯性，确保整体逻辑流畅。
    3. 在报告末尾添加一个**【结论与展望】**部分，总结研究发现并指出未来方向。
    4. 添加一个**【引用来源】**列表，列出所有章节中提到的 URL。
    5. 添加一个**【质量说明】**部分，简要说明各章节的生成迭代过程和关键改进。
    6. 整体报告必须格式优美，使用 Markdown 格式，包含适当的标题层级。
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
