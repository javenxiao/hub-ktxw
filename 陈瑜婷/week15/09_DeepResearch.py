import asyncio
import os
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
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
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-c4395731abd4446b8642c7734c8dbf56")
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
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

reflection_system_prompt = """
你作为一名专业的内容审核员，你需要结合用户的提问和生成的回答，判断回答是否满足了提问的要求。如果满足，则直接回满足，不需要有其他输出。如果不满足，则输出还需要分析的子问题，每行一个。
"""
ReflectionAgent = Agent(
    "Reflection Specialist",
    instructions=reflection_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)
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

    # 3. 并发处理所有章节的检索、抓取和起草
    async def process_section(section: Dict[str, Any], index: int) -> str:
        """处理单个章节：检索、抓取、起草"""
        section_title = section.get("section_title")
        search_keywords = section.get("search_keywords")
        print(f"\n--- Step 3.{index + 1}: 正在处理章节: {section_title} ---")

        # 3.1. 精确检索
        section_query = f"{section_title} 搜索关键词: {search_keywords}"
        section_search_results_str = await async_search_jina(section_query)

        # 3.2. 筛选并抓取前2个链接
        try:
            search_results = json.loads(section_search_results_str)
            urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
        except:
            print("Warning: Failed to parse search results for crawl.")
            urls_to_crawl = []

        crawled_content = []
        for url in urls_to_crawl:
            content = await async_crawl_jina(url)
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

        try:
            section_draft = await Runner.run(
                DraftingAgent,
                draft_prompt,
            )
            result = f"## {section_title}\n\n{section_draft.final_output}"
            print(f"-> 章节起草完成: {section_title}")
            print(section_draft.final_output)
            print("")
            return result
        except Exception as e:
            error_msg = f"章节起草失败: {e}"
            result = f"## {section_title}\n\n{error_msg}"
            print(f"-> {error_msg}")
            return result

    # 使用 asyncio.gather 并发处理所有章节
    section_tasks = [process_section(section, i) for i, section in enumerate(sections)]
    drafted_sections = await asyncio.gather(*section_tasks)

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
        current_report = final_report.final_output
    except Exception as e:
        current_report = f"最终报告整合失败: {e}\n\n已完成的章节草稿:\n{full_report_draft}"

    # 5. 反思机制：评估报告质量并逐步改进
    print("\nStep 5: 启动反思机制，评估报告质量...")
    max_reflection_iterations = 3  # 最大反思迭代次数
    reflection_iteration = 0
    
    while reflection_iteration < max_reflection_iterations:
        reflection_iteration += 1
        print(f"\n--- 反思迭代 {reflection_iteration}/{max_reflection_iterations} ---")
        
        # 5.1. 调用 ReflectionAgent 评估当前报告
        reflection_prompt = f"""
        用户提问: {query}
        
        生成的回答（研究报告）:
        {current_report}
        
        请判断这个研究报告是否满足了用户提问的要求。
        """
        
        try:
            reflection_result = await Runner.run(
                ReflectionAgent,
                reflection_prompt,
            )
            reflection_output = reflection_result.final_output.strip()
            print(f"反思结果: {reflection_output}")
            
            # 如果返回"满足"或类似表示满足的文本，则完成
            if "满足" in reflection_output or len(reflection_output) < 10:
                print("✓ 报告质量评估通过，满足用户要求！")
                break
            
            # 如果不满足，提取子问题列表
            sub_questions = [q.strip() for q in reflection_output.split("\n") if q.strip() and len(q.strip()) > 2]
            
            if not sub_questions:
                print("反思结果为空，认为已满足要求")
                break
            
            print(f"发现 {len(sub_questions)} 个需要进一步分析的问题:")
            for i, q in enumerate(sub_questions, 1):
                print(f"  {i}. {q}")
            
            # 5.2. 对每个子问题进行研究并生成补充内容
            print("\n开始针对反馈问题进行深入研究...")
            supplementary_sections = []
            
            async def process_supplementary_question(sub_question: str, index: int) -> str:
                """处理补充问题：检索、抓取、起草"""
                print(f"\n--- 处理补充问题 {index + 1}: {sub_question} ---")
                
                # 搜索
                search_results_str = await async_search_jina(sub_question)
                
                # 抓取前2个链接
                try:
                    search_results = json.loads(search_results_str)
                    urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
                except:
                    print("Warning: Failed to parse search results for crawl.")
                    urls_to_crawl = []
                
                crawled_content = []
                for url in urls_to_crawl:
                    content = await async_crawl_jina(url)
                    crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")
                
                raw_materials = "\n\n".join(crawled_content)
                
                # 起草补充内容
                draft_prompt = f"""
                **问题:** {sub_question}
                
                **搜索结果摘要:**
                {search_results_str[:3000]}... (仅用于参考要点)
                
                **原始网页内容 (请基于此内容撰写):**
                {raw_materials}
                
                请根据上述信息，针对"{sub_question}"这个问题，撰写详细的补充内容。
                内容应该能够补充和完善现有研究报告。
                """
                
                try:
                    section_draft = await Runner.run(
                        DraftingAgent,
                        draft_prompt,
                    )
                    result = f"## 补充内容: {sub_question}\n\n{section_draft.final_output}"
                    print(f"-> 补充内容起草完成: {sub_question}")
                    return result
                except Exception as e:
                    error_msg = f"补充内容起草失败: {e}"
                    result = f"## 补充内容: {sub_question}\n\n{error_msg}"
                    print(f"-> {error_msg}")
                    return result
            
            # 并发处理所有补充问题
            supplementary_tasks = [process_supplementary_question(q, i) for i, q in enumerate(sub_questions)]
            supplementary_sections = await asyncio.gather(*supplementary_tasks)
            
            # 5.3. 将补充内容整合到报告中
            print("\n整合补充内容到报告中...")
            supplementary_content = "\n\n".join(supplementary_sections)
            
            enhancement_prompt = f"""
            请将以下补充内容整合到现有研究报告中，形成一份更完整、更全面的报告。
            
            **现有报告:**
            {current_report}
            
            **补充内容:**
            {supplementary_content}
            
            **任务要求:**
            1. 将补充内容自然地融入到现有报告中，可以创建新章节或补充到相关章节
            2. 保持报告的整体结构和连贯性
            3. 确保所有内容都围绕原始研究主题: {query}
            4. 保持 Markdown 格式
            """
            
            try:
                enhanced_report = await Runner.run(
                    DeepResearchAgent,
                    enhancement_prompt,
                )
                current_report = enhanced_report.final_output
                print("✓ 补充内容已整合到报告中")
            except Exception as e:
                # 如果整合失败，简单追加补充内容
                current_report = f"{current_report}\n\n--- 补充内容 ---\n\n{supplementary_content}"
                print(f"Warning: 整合失败，直接追加补充内容: {e}")
            
        except Exception as e:
            print(f"反思过程出错: {e}")
            break
    
    if reflection_iteration >= max_reflection_iterations:
        print(f"\n已达到最大反思迭代次数 ({max_reflection_iterations})，返回当前报告")
    
    return current_report

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