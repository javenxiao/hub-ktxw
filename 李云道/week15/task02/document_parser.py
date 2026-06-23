import os
import subprocess
import json
import logging
from typing import Optional, Dict, Any


class DocumentParser:
    """文档解析模块 - 使用MinerU进行PDF/Word解析"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)

    def parse_document(self, file_path: str, method: str = "auto") -> str:
        """
        使用MinerU解析文档

        Args:
            file_path: 输入文档路径
            method: 解析方法 (ocr|txt|auto)

        Returns:
            解析后的文本内容
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文档不存在: {file_path}")

        # 调用MinerU命令行工具
        cmd = [
            "magic-pdf",
            "-p", file_path,
            "-o", self.output_dir,
            "-m", method
        ]

        try:
            self.logger.info(f"开始解析文档: {file_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 读取解析结果
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            md_file = os.path.join(self.output_dir, f"{base_name}.md")

            if os.path.exists(md_file):
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.logger.info(f"文档解析成功，内容长度: {len(content)}")
                return content
            else:
                raise FileNotFoundError(f"解析结果文件未找到: {md_file}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"MinerU解析失败: {e.stderr}")
            raise RuntimeError(f"文档解析错误: {e.stderr}")

    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """获取文档基本信息"""
        return {
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "file_type": os.path.splitext(file_path)[1].lower()
        }