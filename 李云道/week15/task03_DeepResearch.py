import asyncio
import os
from openai import OpenAI
import json
import requests
import urllib.parse
from typing import List, Dict, Any

# 假设以下导入能够正常工作，它们通常来自 agents 库
from agents import (
    Agent,
    function_tool,  # 装饰器，使Agent(tools=[])能调用函数
    AsyncOpenAI,  # 异步客户端，设置API_KEY和API_URL
    OpenAIChatCompletionsModel,
    ModelSettings,  # 模型设置，模型名称、客户端、温度等超参数
    Runner,  # 异步运行器，用于执行 Agent
    set_default_openai_api,  # 聊天对话，使用上下文
    set_tracing_disabled  # 禁用追踪功能，加速
)

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

'''
将 09_DeepResearch.py 改为不同章节同时生成，并且加入 方式/react 的机制，大模型判断这个这个章节的生成效果，有反馈建议，逐步生成
'''

os.environ["OPENAI_API_KEY"] = "sk-2ff484a65dbd47668f71c459353fd8ff"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL_NAME = "qwen-max"  # 假设这是 AliCloud 兼容模式下的一个模型名称
API_KEY = os.getenv("OPENAI_API_KEY", "sk-2ff484a65dbd47668f71c459353fd8ff")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化 AsyncOpenAI 客户端
llm_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 定义模型设置
model_settings = ModelSettings(
    temperature=0.3,
    parallel_tool_calls=False
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

# --- 3.1 代理定义 (Agents) ---
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

# 3.3 react agent
reacting_system_prompt = """
你是一个内容质量评估专家。你的任务是判断给定的章节标题与生成的章节内容是否合理匹配。

## 评估标准：
1. **相关性**：内容是否与标题主题直接相关
2. **完整性**：内容是否覆盖了标题所暗示的各个方面
3. **逻辑性**：内容结构是否合理，逻辑是否连贯
4. **专业性**：内容是否具备应有的专业深度和准确性

## 输入格式：
- section_title: 章节标题字符串
- section_draft: 生成的章节内容

## 输出要求：
- 只输出单个词语：True 或 False
- True表示合理匹配，False表示不合理匹配
- 不要添加任何解释、标点或额外文本

## 判断指南：
输出True的情况：
- 内容与标题高度相关且完整
- 虽有小瑕疵但整体合理
- 专业术语使用恰当

输出False的情况：
- 内容与标题完全无关
- 关键信息缺失严重
- 存在明显事实错误
- 逻辑混乱无法理解

请严格遵循输出格式，只返回True或False。
"""
ReactAgent = Agent(
    "",
    instructions=reacting_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)


# --- 4. 深度研究核心流程 ---
async def deep_research_step1_plan(query: str, max_sections: int = 5) -> tuple[str, list[dict[str:str]]]:
    '''
    规划。先确定主题，然后规划章节
    :param query:
    :param max_sections:
    :return:
    '''
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
    sections: list[dict[str:str]] = outline_json.get("sections", [])
    if len(sections) > max_sections:
        sections = sections[:max_sections]

    print(f"报告标题: {research_title}")
    print(f"规划了 {len(sections)} 个章节。")
    return research_title, sections


async def deep_research_step2_retrieve_and_draft(sections: list[dict[str:str]]):
    """并行处理所有章节"""

    async def evaluate_section(section_title, section_draft):
        """
        评估章节标题与内容的合理性

        Args:
            section_title: 章节标题
            section_draft: 章节草稿对象，包含final_output属性

        Returns:
            bool: 是否合理
        """
        try:
            # 构造评估提示
            evaluation_prompt = f"""
            section_title: {section_title}
            section_draft: {section_draft.final_output if hasattr(section_draft, 'final_output') else str(section_draft)}

            请评估上述章节标题与内容是否合理匹配。
            """

            # 调用ReactAgent进行评估
            evaluation_result = await Runner.run(
                ReactAgent,
                evaluation_prompt,
            )

            # 清理输出并转换为布尔值
            raw_output = evaluation_result.final_output.strip().lower()

            # 处理各种可能的输出格式
            if raw_output == "true":
                is_reasonable = True
            elif raw_output == "false":
                is_reasonable = False
            elif "true" in raw_output:
                is_reasonable = True
            elif "false" in raw_output:
                is_reasonable = False
            else:
                # 默认处理：如果代理没有按格式输出，根据内容长度等启发式判断
                content = section_draft.final_output if hasattr(section_draft, 'final_output') else str(section_draft)
                is_reasonable = len(content) > 50  # 简单启发式规则

            print(f"章节评估结果: {section_title} -> {is_reasonable}")
            return is_reasonable

        except Exception as e:
            print(f"章节评估失败: {e}")
            # 出错时默认返回False以确保质量
            return False

    # 处理单章节
    async def process_section(section, i):
        """处理单个章节的异步函数"""
        section_title = section.get("section_title")
        search_keywords = section.get("search_keywords")
        print(f"\n--- 并行处理章节 {i + 1}: {section_title} ---")

        # 3.1. 精确检索
        section_query = f"{section_title} 搜索关键词: {search_keywords}"
        section_search_results_str = await async_search_jina(section_query)

        # 3.2. 筛选并抓取前2个链接
        try:
            search_results = json.loads(section_search_results_str)
            urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
        except:
            print(f"Warning: Failed to parse search results for chapter {i + 1}")
            urls_to_crawl = []

        # 并行抓取多个URL
        crawl_tasks = []
        for url in urls_to_crawl:
            task = asyncio.create_task(async_crawl_jina(url))
            crawl_tasks.append((url, task))

        crawled_content = []
        for url, task in crawl_tasks:
            try:
                content = await task
                crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")
            except Exception as e:
                print(f"Failed to crawl {url}: {e}")

        raw_materials = "\n\n".join(crawled_content)

        is_reasonable = False
        section_draft_content = ""
        retry_count = 10

        while not is_reasonable and retry_count > 0:

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
                print(f"-> Task.{i}-Retry.{11 - retry_count} 章节起草完成: {section_title}")
                # return f"## {section_title}\n\n{section_draft.final_output}"
                section_draft_content = section_draft.final_output
            except Exception as e:
                error_msg = f"章节起草失败: {e}"
                print(f"-> Task.{i}-Retry.{11 - retry_count} 发生错误: {error_msg}")
                # return f"## {section_title}\n\n{error_msg}"
                section_draft_content = error_msg
            finally:
                is_reasonable = await evaluate_section(section_title, section_draft_content)
                print(f"-> Task.{i}-Retry.{11 - retry_count} 起草合理性 {is_reasonable}")

        return section_draft_content

    # 创建所有章节的处理任务
    tasks = []
    for i, section in enumerate(sections):
        task = asyncio.create_task(process_section(section, i))
        tasks.append(task)

    # 使用asyncio.gather同时执行所有任务[6,8](@ref)
    drafted_sections = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理可能出现的异常
    final_sections = []
    for i, result in enumerate(drafted_sections):
        if isinstance(result, Exception):
            error_msg = f"## {sections[i].get('section_title')}\n\n处理时发生错误: {result}"
            final_sections.append(error_msg)
            print(f"章节 {i + 1} 处理失败: {result}")
        else:
            final_sections.append(result)

    return final_sections


async def deep_research_step3_content_integrate(research_title: str, drafted_sections: list[str]):
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
        4. 添加一个**【引用来源】**列表放在文档最后，列出所有章节中提到的 URL。
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


async def deep_research(query: str) -> str:
    """
    执行深度研究流程：规划 -> 检索 -> 抓取 -> 起草 -> 整合。
    """
    print(f"\n--- Deep Research for: {query} ---\n")

    # 规划
    research_title, sections = await deep_research_step1_plan(query, max_sections=5)

    # 逐章进行检索、抓取和起草
    drafted_sections = await deep_research_step2_retrieve_and_draft(sections)

    # 报告整合与最终输出
    final_report = await deep_research_step3_content_integrate(research_title, drafted_sections)

    return final_report


async def main():
    research_topic = "Agentic AI在软件开发中的最新应用和挑战"
    final_report = await deep_research(research_topic)
    print(final_report)


async def for_test():
    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-2ff484a65dbd47668f71c459353fd8ff"
    )
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": "You are an assistant。"},
            {"role": "user", "content": "nihao"}
        ],
        max_tokens=1000,
        temperature=0.3,
        stream=False,
    )
    print(response.choices[0].message.content)


# 使用 Runner 启动异步主函数
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except NameError:
        # Fallback to standard asyncio run if Runner is not defined or preferred
        asyncio.run(main())
        # pass
