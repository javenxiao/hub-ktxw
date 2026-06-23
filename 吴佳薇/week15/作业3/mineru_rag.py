import os
import subprocess
import tempfile
from pathlib import Path


# ===== 1. 使用mineru命令行解析PDF =====
def parse_pdf_with_mineru(pdf_path):
    """使用mineru命令行工具解析PDF"""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # 构建并执行mineru命令
        cmd = ["mineru", pdf_path, "-o", str(output_dir)]
        subprocess.run(cmd, check=True)

        # 读取解析结果
        for ext in ["*.txt", "*.md"]:
            result_files = list(output_dir.glob(ext))
            if result_files:
                with open(result_files[0], 'r', encoding='utf-8') as f:
                    return f.read()

    return ""


# ===== 2. RAG问答核心功能（使用qwen-max）=====
def setup_rag_qa_system(pdf_text):
    """基于解析文本构建RAG问答系统"""

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(pdf_text)
    print(f"文本分割为 {len(chunks)} 个片段")

    # 创建向量数据库
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 初始化通义千问 qwen-max
    llm = ChatOpenAI(
        model="qwen-max",
        openai_api_key="sk-6512723ce39a49f28a68e69b6ace52ab",  #已脱敏处理
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1,
        max_tokens=2048
    )

    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain


# ===== 3. 主流程 =====
def main():
    # 配置参数
    pdf_file = "东方证券-张庆-大语言模型在投研投顾中的应用与难点.pdf"

    # 步骤1: 解析PDF
    print("解析PDF中...")
    pdf_text = parse_pdf_with_mineru(pdf_file)
    print(f"解析完成，获取字符数: {len(pdf_text)}")

    # 步骤2: 构建RAG系统
    print("构建RAG系统中...")
    qa_system = setup_rag_qa_system(pdf_text)

    # 步骤3: 示例问答
    print("\n--- 开始问答 ---")
    test_questions = [
        "这份文档主要讲了什么？",
        "大语言模型在投研投顾中有哪些应用？",
        "文档中提到了哪些挑战或难点？"
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        answer = qa_system.run(question)
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
