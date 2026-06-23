from doc_retriever import DocSummaryRetriever
import time
from multiprocessing import freeze_support
from generate_mcp_tool import generate_mcp_tool_from_md as gen_mcp_tool
from model_mcp_client import demo_generated_model

query = "用户的问题"
md_docments_folder_path=r"D:\438161609\WPS云盘\BADOU\week14\Week14\07-文档公式解析与智能问答\minerU_pdf2md\extracted_10"
if __name__ == "__main__":
    freeze_support()
    # 1.根据用户提问用LLM生成文件夹内匹配的md文档
    start_time = time.time()
    print("程序开始运行...")
    # 初始化检索器
    retriever = DocSummaryRetriever()
    # 从文件夹构建索引（首次运行或文件更新时执行）
    retriever.build_index_from_folder(md_docments_folder_path)  # 替换为你的MD文件文件夹路径
    # 搜索相关文件
    relevant_files = retriever.search_relevant_files(query)
    print("\n相关文件：")
    for file in relevant_files:
        print(f"- {file['filename']}（相关性：{file['relevance_score']:.2f}）: {file['summary']}")

    # 搜索并返回完整内容
    full_results = retriever.search_with_source(query)
    print("\n完整结果：")
    for res in full_results:
        print(f"\n来源：{res['source']}")
        print(f"摘要：{res['summary']}")
        print(f"内容：{res['content'][:500]}...")  # 打印前500字

    # 计算并打印总耗时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n生成摘要并检索总耗时: {total_time:.2f} 秒")

    #2.根据检索的md文档生成mcp_tool代码
    gen_mcp_tool(
        r"D:\438161609\WPS云盘\BADOU\week14\Week14\07-文档公式解析与智能问答\minerU_pdf2md\extracted_10\0ba15b17-85d2-4944-9a04-a9bd23c2e3f.md",
        r"D:\438161609\WPS云盘\BADOU\week14\Week14\07-文档公式解析与智能问答\minerU_pdf2md\rag\gen_mcp_tool")
    # 3.运行生成的mcp_tool并调用演示
    import asyncio
    asyncio.run(demo_generated_model())
