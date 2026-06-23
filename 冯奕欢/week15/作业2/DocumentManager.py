from pathlib import Path
from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import re


class MarkdownDocumentManager:

    def __init__(self, directory_path, glob_pattern="./*.md"):
        self.directory_path = Path(directory_path)
        self.glob_pattern = glob_pattern
        self.processed_documents = []  # 预处理后文档（HTML转MD表格）
        self.all_chunks = []  # 最终拆分的Chunk

    # --------------------------
    # 通用工具函数：HTML表格转标准MD表格
    # --------------------------
    def html_table_to_md(self, html_content: str) -> str:
        """通用HTML表格转MD表格：适配任意HTML表格结构"""
        soup = BeautifulSoup(html_content, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return html_content

        for table in tables:
            md_table = "\n"
            rows = table.find_all("tr")
            if not rows:
                continue

            # 通用提取单元格（兼容<th>/<td>，无业务逻辑）
            all_cells = []
            for row in rows:
                cells = [
                    td.get_text(strip=True).replace("\n", " ").replace("  ", " ")
                    for td in row.find_all(["th", "td"])
                ]
                if cells:
                    all_cells.append(cells)

            # 生成通用MD表格（无业务拆分）
            if all_cells:
                # 表头行
                md_table += "| " + " | ".join(all_cells[0]) + " |\n"
                # 分隔线
                md_table += "| " + " | ".join(["---"] * len(all_cells[0])) + " |\n"
                # 数据行
                for row_cells in all_cells[1:]:
                    # 补全列数（兼容不规则表格）
                    row_cells += [""] * (len(all_cells[0]) - len(row_cells))
                    md_table += "| " + " | ".join(row_cells) + " |\n"

            # 替换原HTML表格为通用MD表格
            html_content = html_content.replace(str(table), md_table)
        return html_content

    # --------------------------
    # 通用工具函数：提取MD文档中的所有完整表格
    # --------------------------
    def extract_all_tables(self, md_text: str) -> (str, list):
        """
        通用提取所有完整MD表格：
        返回：去除表格的纯文本、完整表格Chunk列表
        """
        # 通用MD表格正则（匹配|开头的表格行+分隔线）
        table_pattern = r"(?:\|.*\|\n){2,}"
        # 查找所有表格（非贪婪匹配，避免跨表格合并）
        tables = re.findall(table_pattern, md_text, re.DOTALL)
        table_chunks = []

        if tables:
            # 为每个完整表格生成独立Chunk（通用规则）
            for idx, table in enumerate(tables, 1):
                clean_table = table.strip()
                table_chunks.append(Document(
                    page_content=f"### 表格 {idx}\n{clean_table}",
                    metadata={"type": "table", "table_index": idx}
                ))
            # 移除原文本中的表格（保留纯文本）
            md_text_without_table = re.sub(table_pattern, "", md_text, re.DOTALL).strip()
        else:
            md_text_without_table = md_text

        return md_text_without_table, table_chunks

    # --------------------------
    # 核心步骤1：通用加载+预处理（适配任意MD文档）
    # --------------------------
    def load_and_preprocess(self):
        md_files = list(self.directory_path.glob(self.glob_pattern))
        for file in md_files:
            try:
                # 通用读取（兼容UTF-8/GBK编码）
                try:
                    raw_content = file.read_text(encoding="utf-8")
                except:
                    raw_content = file.read_text(encoding="gbk")

                # 通用预处理：HTML转MD表格 + 清理特殊字符
                processed_content = self.html_table_to_md(raw_content)
                processed_content = processed_content.replace("\r", "\n") \
                    .replace("\u3000", " ") \
                    .replace("  ", " ") \
                    .strip()

                self.processed_documents.append(Document(
                    page_content=processed_content,
                    metadata={"source": str(file), "file_name": file.name}
                ))
                print(f"✅ 通用预处理完成：{file.name}")
            except Exception as e:
                print(f"❌ 处理失败 {file.name}：{str(e)}")
        print(f"\n📊 总计预处理文档数：{len(self.processed_documents)}")

    # --------------------------
    # 核心步骤2：通用拆分（标题文本+完整表格）
    # --------------------------
    def split_all_documents(self):
        # 通用标题拆分规则（适配任意MD标题层级）
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4")
        ]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        for doc in self.processed_documents:
            # 1. 通用提取完整表格为独立Chunk
            text_without_table, table_chunks = self.extract_all_tables(doc.page_content)

            # 2. 通用拆分标题文本（无业务逻辑）
            text_sections = text_splitter.split_text(text_without_table)
            # 为文本Chunk补充通用元数据
            for section in text_sections:
                section.metadata.update({
                    "source": doc.metadata["source"],
                    "file_name": doc.metadata["file_name"],
                    "type": "text"
                })

            # 3. 为表格Chunk补充通用元数据
            for table_chunk in table_chunks:
                table_chunk.metadata.update({
                    "source": doc.metadata["source"],
                    "file_name": doc.metadata["file_name"]
                })

            # 4. 合并所有Chunk（文本+表格）
            self.all_chunks.extend(text_sections)
            self.all_chunks.extend(table_chunks)

        # 通用统计
        text_chunk_count = len([c for c in self.all_chunks if c.metadata["type"] == "text"])
        table_chunk_count = len([c for c in self.all_chunks if c.metadata["type"] == "table"])
        print(f"📊 总计生成通用Chunk数：{len(self.all_chunks)}")
        print(f"   - 文本Chunk数：{text_chunk_count}")
        print(f"   - 完整表格Chunk数：{table_chunk_count}")