#!/usr/bin/env python3
"""
RAG系统主入口模块
使用MinerU解析文档 + FAISS向量检索 + LLM生成答案
"""

import os
import argparse
import logging
import sys
from document_parser import DocumentParser
from text_processor import TextProcessor
from vector_indexer import VectorIndexer
from rag_engine import RAGEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """完整的RAG系统主类"""

    def __init__(self, api_key: str = None):
        self.document_parser = DocumentParser()
        self.text_processor = TextProcessor()
        self.vector_indexer = VectorIndexer()
        self.rag_engine = RAGEngine(api_key=api_key)
        self.is_index_built = False

    def process_document(self, file_path: str, rebuild_index: bool = True) -> bool:
        """
        处理文档并构建索引

        Args:
            file_path: 文档路径
            rebuild_index: 是否重新构建索引

        Returns:
            处理是否成功
        """
        try:
            logger.info(f"开始处理文档: {file_path}")

            # 1. 解析文档
            parsed_content = self.document_parser.parse_document(file_path)
            if not parsed_content:
                logger.error("文档解析失败")
                return False

            # 2. 处理文本
            chunks = self.text_processor.split_text(parsed_content)
            if not chunks:
                logger.error("文本处理失败")
                return False

            # 3. 构建向量索引
            if rebuild_index:
                self.vector_indexer.build_index(chunks)
                self.is_index_built = True
                logger.info("向量索引构建完成")

            return True

        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            return False

    def query_document(self, question: str, k: int = 3) -> dict:
        """查询文档"""
        if not self.is_index_built:
            return {"error": "请先处理文档构建索引"}

        try:
            result = self.rag_engine.rag_query(question, self.vector_indexer, k)
            return result
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {"error": f"查询处理失败: {str(e)}"}

    def interactive_mode(self):
        """交互式问答模式"""
        if not self.is_index_built:
            print("请先使用 process_document() 方法处理文档")
            return

        print("=== RAG文档问答系统 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("-" * 40)

        while True:
            try:
                question = input("\n请输入问题: ").strip()

                if question.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用！")
                    break

                if not question:
                    continue

                print("思考中...")
                result = self.query_document(question)

                if "error" in result:
                    print(f"错误: {result['error']}")
                else:
                    print(f"\n答案: {result['answer']}")
                    print(f"检索到 {result['retrieval_count']} 个相关段落")

            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"处理错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG文档问答系统")
    parser.add_argument("--file", "-f", help="要处理的文档路径")
    parser.add_argument("--question", "-q", help="要查询的问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")

    args = parser.parse_args()
    # args.file = os.path.abspath(r"D:\2025.0717BDLearnAI\FastNLP\homework-new\submit\李云道\week14\pdf\0daef473-e660-4984-be4d-940433aa889.pdf")
    # args.interactive = True

    # 初始化系统
    rag_system = RAGSystem()

    if args.file:
        # 处理文档模式
        if not os.path.exists(args.file):
            print(f"错误: 文件不存在 {args.file}")
            return 1

        success = rag_system.process_document(args.file)
        if not success:
            print("文档处理失败")
            return 1

        print("文档处理完成！")

        if args.question:
            # 单次查询
            result = rag_system.query_document(args.question)
            if "error" in result:
                print(f"错误: {result['error']}")
            else:
                print(f"问题: {result['question']}")
                print(f"答案: {result['answer']}")
        elif args.interactive:
            # 交互模式
            rag_system.interactive_mode()
        else:
            print("使用 --question 参数提问或 --interactive 进入交互模式")

    else:
        print("请使用 --file 参数指定要处理的文档")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())