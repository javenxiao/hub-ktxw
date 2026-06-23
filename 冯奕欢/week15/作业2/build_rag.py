from DocumentManager import MarkdownDocumentManager
from EmbeddingManager import ChromaEmbeddingManager

if __name__ == '__main__':
    # 初始化并加载拆分文档
    dm = MarkdownDocumentManager(directory_path="./md")  # 替换为你的MD目录
    print("加载文件")
    dm.load_and_preprocess()
    print("解析文件")
    dm.split_all_documents()
    # 打印验证
    for chunk in dm.all_chunks:
        print(f"标题元数据：{chunk.metadata}")
        print(f"内容：{chunk.page_content}")
    # 保存为知识库
    em = ChromaEmbeddingManager()
    em.save_chunks_to_chroma(dm.all_chunks)