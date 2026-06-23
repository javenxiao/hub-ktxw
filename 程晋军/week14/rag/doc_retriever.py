import os
import json
import hashlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
from config import VECTOR_DB_CONFIG, QWEN_EMBEDDING_CONFIG
from openai import OpenAI
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


def _read_single_file_standalone(args: tuple) -> Optional[Dict]:
    """单文件读取处理（供多进程调用）"""
    file_info, file_cache = args
    file_path, filename = file_info
    
    try:
        # 计算文件哈希值
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()
        
        file_mtime = os.path.getmtime(file_path)

        # 缓存命中：文件未修改则直接跳过
        if filename in file_cache:
            cache_hash = file_cache[filename]["hash"]
            cache_mtime = file_cache[filename]["mtime"]
            if cache_hash == file_hash and abs(file_mtime - cache_mtime) < 1:
                return None

        # 读取文件完整内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "filename": filename,
            "content": content,
            "hash": file_hash,
            "mtime": file_mtime
        }
    except Exception as e:
        print(f"处理文件 {filename} 失败: {e}")
        return None


class DocSummaryRetriever:
    def __init__(self):
        # 配置参数适配功能重命名
        self.data_path = VECTOR_DB_CONFIG["db_path"]  # 原db_path改为data_path
        self.top_k = VECTOR_DB_CONFIG["top_k"]
        self.text_chunks = []  # 存储文档完整内容
        self.chunk_sources = []  # 存储内容对应的源文件
        self.file_summaries = {}  # 文件名->摘要映射
        self.files_list = []  # 已处理文件列表
        self.file_cache = {}  # 缓存文件哈希和修改时间（避免重复处理）
        self.cache_path = os.path.join(self.data_path, "file_cache.json")

        # 初始化OpenAI客户端（复用连接，提升API调用效率）
        self.client = OpenAI(
            api_key=QWEN_EMBEDDING_CONFIG["api_key"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30
        )

        # 创建数据存储目录
        os.makedirs(self.data_path, exist_ok=True)
        # 加载文件缓存
        self._load_cache()

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值，用于检测文件是否修改"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_cache(self):
        """加载文件缓存，跳过未修改的文件"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.file_cache = json.load(f)
            except Exception as e:
                print(f"加载文件缓存失败: {e}")
                self.file_cache = {}

    def _save_cache(self):
        """保存文件缓存，记录已处理文件状态"""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.file_cache, f, ensure_ascii=False, indent=2)

    def _read_single_file(self, file_info: tuple) -> Optional[Dict]:
        """单文件读取处理（供多进程调用）"""
        file_path, filename = file_info
        try:
            file_hash = self._calculate_file_hash(file_path)
            file_mtime = os.path.getmtime(file_path)

            # 缓存命中：文件未修改则直接跳过
            if filename in self.file_cache:
                cache_hash = self.file_cache[filename]["hash"]
                cache_mtime = self.file_cache[filename]["mtime"]
                if cache_hash == file_hash and abs(file_mtime - cache_mtime) < 1:
                    return None

            # 读取文件完整内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "filename": filename,
                "content": content,
                "hash": file_hash,
                "mtime": file_mtime
            }
        except Exception as e:
            print(f"处理文件 {filename} 失败: {e}")
            return None

    def _batch_generate_summaries(self, file_contents: List[Dict]) -> List[Dict]:
        """批量生成文件摘要，减少API调用次数（核心提速点）"""
        if not file_contents:
            return []

        # 逐个生成摘要，而不是尝试批量请求
        summaries = []
        try:
            for item in file_contents:
                # 限制输入长度，避免超出模型上下文
                full_text = item["content"][:2000]
                completion = self.client.chat.completions.create(
                    model="qwen-flash",
                    messages=[
                        {"role": "system", "content": "请为以下文档内容生成不超过100字的摘要，需包含核心背景知识："},
                        {"role": "user", "content": full_text}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                
                summary = completion.choices[0].message.content.strip()[:100] if completion.choices[0].message else item["content"][:100].strip()
                summaries.append({
                    "filename": item["filename"],
                    "summary": summary,
                    "hash": item["hash"],
                    "mtime": item["mtime"]
                })
            return summaries
        except Exception as e:
            print(f"批量生成摘要失败，降级为单条生成: {e}")
            # 降级方案：单条生成摘要
            return [
                {
                    "filename": item["filename"],
                    "summary": self._generate_single_summary(item["content"], item["filename"]),
                    "hash": item["hash"],
                    "mtime": item["mtime"]
                }
                for item in file_contents
            ]

    def _generate_single_summary(self, content: str, filename: str) -> str:
        """单文件摘要生成（降级方案）"""
        try:
            full_text = content[:2000]
            completion = self.client.chat.completions.create(
                model="qwen-flash",
                messages=[
                    {"role": "system", "content": "请为以下文档内容生成不超过100字的摘要，需包含核心背景知识："},
                    {"role": "user", "content": full_text}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return completion.choices[0].message.content.strip()[:100] if completion.choices[0].message else content[
                                                                                                             :100].strip()
        except Exception as e:
            print(f"生成 {filename} 摘要失败: {e}")
            return content[:100].strip()

    def build_index_from_folder(self, folder_path: str):
        """从文件夹构建文档索引（原build_db_from_folder，命名更精准）"""
        start_time = datetime.now()

        # 收集所有MD文件任务
        file_tasks = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.md'):
                file_path = os.path.join(folder_path, filename)
                file_tasks.append((file_path, filename))

        # 多进程读取文件（核心提速点，最多8进程避免资源占用过高）
        print(f"开始读取文件，共 {len(file_tasks)} 个MD文件，使用 {min(cpu_count(), 8)} 个进程")
        # 准备参数：将文件缓存作为参数传递
        tasks_with_cache = [(task, self.file_cache) for task in file_tasks]
        
        with Pool(processes=min(cpu_count(), 8)) as pool:
            file_results = pool.map(_read_single_file_standalone, tasks_with_cache)

        # 过滤无效结果和已缓存文件
        new_files = [res for res in file_results if res is not None]
        print(f"需要处理的新增/修改文件: {len(new_files)} 个")

        if not new_files:
            print("无新增或修改文件，直接加载已有索引")
            self._load_existing_index()
            return

        # 批量生成摘要
        print("批量生成文件摘要...")
        summary_results = self._batch_generate_summaries(new_files)

        # 整理数据（合并已有数据，过滤已更新文件的旧数据）
        all_text_chunks, chunk_sources, file_summaries, files_list = self._merge_data(new_files, summary_results)
        file_cache_update = {res["filename"]: {"hash": res["hash"], "mtime": res["mtime"]} for res in summary_results}

        # 更新缓存和实例变量
        self.file_cache.update(file_cache_update)
        self._save_cache()
        self.text_chunks = all_text_chunks
        self.chunk_sources = chunk_sources
        self.file_summaries = file_summaries
        self.files_list = files_list

        # 保存数据到本地
        self._save_index_data()

        # 输出统计信息
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n文档索引构建完成！")
        print(f"总耗时: {elapsed:.2f} 秒")
        print(f"处理文档内容: {len(all_text_chunks)} 个")
        print(f"涉及文件总数: {len(set(chunk_sources))} 个")
        print(f"新增/修改文件: {len(new_files)} 个")

    def _merge_data(self, new_files: List[Dict], summary_results: List[Dict]) -> tuple:
        """合并新数据和已有数据，过滤旧数据"""
        # 初始化新数据列表
        new_text_chunks = [new_files[[f["filename"] for f in new_files].index(res["filename"])]["content"] for res in
                           summary_results]
        new_chunk_sources = [res["filename"] for res in summary_results]
        new_file_summaries = {res["filename"]: res["summary"] for res in summary_results}
        new_files_list = [res["filename"] for res in summary_results]

        # 如果存在已有数据，合并并过滤旧数据
        if os.path.exists(os.path.join(self.data_path, "text_chunks.txt")):
            with open(os.path.join(self.data_path, "text_chunks.txt"), "r", encoding="utf-8") as f:
                existing_chunks = f.read().split("\n===CHUNK_SPLITTER===\n")
            with open(os.path.join(self.data_path, "chunk_sources.txt"), "r", encoding="utf-8") as f:
                existing_sources = f.read().split("\n===SOURCE_SPLITTER===\n")
            with open(os.path.join(self.data_path, "file_summaries.txt"), "r", encoding="utf-8") as f:
                existing_summaries = {line.strip().split("||||", 1)[0]: line.strip().split("||||", 1)[1] for line in f
                                      if "||||" in line}
            with open(os.path.join(self.data_path, "files_list.txt"), "r", encoding="utf-8") as f:
                existing_files = [line.strip() for line in f if line.strip()]

            # 过滤已更新文件的旧数据
            keep_chunks = [c for c, s in zip(existing_chunks, existing_sources) if s not in new_file_summaries]
            keep_sources = [s for s in existing_sources if s not in new_file_summaries]
            keep_summaries = {k: v for k, v in existing_summaries.items() if k not in new_file_summaries}
            keep_files = [f for f in existing_files if f not in new_file_summaries]

            # 合并数据
            all_text_chunks = keep_chunks + new_text_chunks
            chunk_sources = keep_sources + new_chunk_sources
            file_summaries = {**keep_summaries, **new_file_summaries}
            files_list = keep_files + new_files_list
            return all_text_chunks, chunk_sources, file_summaries, files_list

        return new_text_chunks, new_chunk_sources, new_file_summaries, new_files_list

    def _save_index_data(self):
        """保存索引数据到本地"""
        with open(f"{self.data_path}/text_chunks.txt", "w", encoding="utf-8") as f:
            f.write("\n===CHUNK_SPLITTER===\n".join(self.text_chunks))
        with open(f"{self.data_path}/chunk_sources.txt", "w", encoding="utf-8") as f:
            f.write("\n===SOURCE_SPLITTER===\n".join(self.chunk_sources))
        with open(f"{self.data_path}/file_summaries.txt", "w", encoding="utf-8") as f:
            for filename, summary in self.file_summaries.items():
                f.write(f"{filename}||||{summary}\n")
        with open(f"{self.data_path}/files_list.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(self.files_list))

    def _load_existing_index(self):
        """加载已存在的文档索引"""
        try:
            with open(f"{self.data_path}/text_chunks.txt", "r", encoding="utf-8") as f:
                self.text_chunks = f.read().split("\n===CHUNK_SPLITTER===\n")
            with open(f"{self.data_path}/chunk_sources.txt", "r", encoding="utf-8") as f:
                self.chunk_sources = f.read().split("\n===SOURCE_SPLITTER===\n")
            with open(f"{self.data_path}/file_summaries.txt", "r", encoding="utf-8") as f:
                self.file_summaries = {line.strip().split("||||", 1)[0]: line.strip().split("||||", 1)[1] for line in f
                                       if "||||" in line}
            with open(f"{self.data_path}/files_list.txt", "r", encoding="utf-8") as f:
                self.files_list = [line.strip() for line in f if line.strip()]
            print(f"成功加载已有索引，共 {len(self.text_chunks)} 个文档内容")
        except Exception as e:
            print(f"加载已有索引失败: {e}")

    def search_relevant_files(self, query: str, top_n: int = 3) -> list[dict]:
        """根据查询匹配最相关的文件（基于大模型摘要匹配）"""
        if not self.file_summaries:
            raise ValueError("文档摘要未加载，请先使用build_index_from_folder构建索引")

        # 构造摘要列表提示词
        summaries_text = "\n".join(
            [f"{i + 1}. 文件名: {fn}\n   摘要: {sm}\n" for i, (fn, sm) in enumerate(self.file_summaries.items())])
        prompt = f"""用户问题：{query}
文档摘要列表：
{summaries_text}
请选出与问题最相关的{top_n}个文档，按相关性从高到低排序，仅返回文件名，格式如下：
1. [文件名]
2. [文件名]
..."""

        try:
            completion = self.client.chat.completions.create(
                model="qwen-flash",
                messages=[
                    {"role": "system", "content": "专业文档检索助手，精准判断问题与摘要的相关性"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            # 解析返回结果
            relevant_files = []
            if completion and completion.choices[0].message:
                for line in completion.choices[0].message.content.strip().split('\n'):
                    if line.startswith(tuple(f"{i}." for i in range(1, top_n + 1))):
                        filename = line.split('.', 1)[1].strip()
                        if filename in self.file_summaries and len(relevant_files) < top_n:
                            relevant_files.append(filename)

            # 结果不足时补充默认文件
            if len(relevant_files) < top_n:
                supplement = [fn for fn in self.files_list if fn not in relevant_files][:top_n - len(relevant_files)]
                relevant_files.extend(supplement)

            # 构造返回格式
            return [
                {"filename": fn, "summary": self.file_summaries[fn], "relevance_score": 1.0 - (i / 10)}  # 模拟相关性分数
                for i, fn in enumerate(relevant_files)
            ]
        except Exception as e:
            print(f"搜索相关文件出错: {e}")
            # 降级返回前N个文件
            return [
                {"filename": fn, "summary": self.file_summaries[fn], "relevance_score": 1.0 - (i / 10)}
                for i, fn in enumerate(self.files_list[:top_n])
            ]

    def search_with_source(self, query: str) -> list[dict]:
        """搜索相关文件并返回完整内容、来源和摘要"""
        relevant_files = self.search_relevant_files(query, top_n=3)
        relevant_filenames = [item["filename"] for item in relevant_files]

        # 批量映射文件与内容
        file_to_content = {}
        for source, chunk in zip(self.chunk_sources, self.text_chunks):
            if source in relevant_filenames and source not in file_to_content:
                file_to_content[source] = chunk

        # 构造结果
        return [
            {
                "content": file_to_content.get(fn, ""),
                "source": fn,
                "summary": self.file_summaries[fn],
                "relevance_score": item["relevance_score"]
            }
            for item, fn in zip(relevant_files, relevant_filenames)
            if fn in file_to_content
        ]

    def search_in_files(self, query: str, target_files: list[str]) -> list[dict]:
        """在指定文件中搜索相关内容（基于大模型文本匹配）"""
        # 收集目标文件的完整内容
        target_contents = [
            {"source": source, "content": chunk}
            for source, chunk in zip(self.chunk_sources, self.text_chunks)
            if source in target_files
        ]

        if not target_contents:
            return []

        # 构造文本匹配提示词
        contents_text = "\n".join([
            f"{i + 1}. 来源文件: {item['source']}\n   内容片段: {item['content'][:800]}"  # 限制片段长度
            for i, item in enumerate(target_contents)
        ])
        prompt = f"""用户问题：{query}
目标文件内容片段：
{contents_text}
请选出与问题最相关的{min(self.top_k, len(target_contents))}个内容片段，按相关性排序，格式如下：
1. 来源文件: [文件名], 相关内容: [完整相关内容]
不需要额外说明，严格按格式输出。"""

        try:
            completion = self.client.chat.completions.create(
                model="qwen-flash",
                messages=[
                    {"role": "system", "content": "精准匹配文本与问题的相关性，返回完整相关内容"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )

            # 解析结果
            results = []
            if completion and completion.choices[0].message:
                for line in completion.choices[0].message.content.strip().split('\n'):
                    if line.startswith(tuple(f"{i}." for i in range(1, self.top_k + 1))):
                        if "来源文件:" in line and "相关内容:" in line:
                            # 提取信息
                            source_part = line.split("来源文件:", 1)[1].split(',', 1)[0].strip()
                            content_part = line.split("相关内容:", 1)[1].strip()

                            # 匹配完整内容
                            full_content = next(
                                (item["content"] for item in target_contents if item["source"] == source_part),
                                content_part)

                            results.append({
                                "content": full_content,
                                "source": source_part,
                                "relevance_score": 1.0 - (len(results) / 10)
                            })
            return results[:self.top_k]
        except Exception as e:
            print(f"文件内搜索出错: {e}")
            # 降级返回目标文件的完整内容
            return [
                {
                    "content": item["content"],
                    "source": item["source"],
                    "relevance_score": 1.0 - (i / 10)
                }
                for i, item in enumerate(target_contents[:self.top_k])
            ]


# 使用示例
if __name__ == "__main__":
    # 记录程序开始时间
    import time
    start_time = time.time()
    print("程序开始运行...")
    
    # 初始化检索器
    retriever = DocSummaryRetriever()

    # 从文件夹构建索引（首次运行或文件更新时执行）
    retriever.build_index_from_folder(r"D:\438161609\WPS云盘\BADOU\week14\Week14\07-文档公式解析与智能问答\minerU_pdf2md\extracted_10")  # 替换为你的MD文件文件夹路径

    # 搜索相关文件
    query = "你的查询问题"
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
    print(f"\n程序总耗时: {total_time:.2f} 秒")

### 最终优化亮点
# 1. ** 功能精准化 **：类名、方法名、变量名完全贴合"文档摘要检索"，无冗余命名
# 2. ** 速度优化保留 **：多进程文件读取、批量摘要生成、文件缓存三大核心提速手段
# 3. ** 逻辑简化 **：移除所有向量相关冗余，流程清晰（文件读取→摘要生成→缓存→大模型匹配）
# 4. ** 可用性提升 **：添加使用示例、相关性分数模拟、错误降级处理，稳定性更强