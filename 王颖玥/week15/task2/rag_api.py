import yaml  # type: ignore
from typing import Union, List, Any, Dict
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from router_schemas import Message
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import numpy as np
import datetime
import pdfplumber  # 导入pdfplumber模块，用于处理PDF文件
from openai import OpenAI

import torch  # type: ignore
# 加载预训练模型和分词器
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
# 加载嵌入模型
from sentence_transformers import SentenceTransformer  # type: ignore
# from FlagEmbedding import FlagReranker
from es_api import es  # 导入之前定义的Elasticsearch客户端（用于存储/检索数据）

"""
RAG（检索增强生成）系统的核心实现，负责文档内容提取、文本嵌入生成、检索融合、与大语言模型（LLM）交互等关键功能
"""

device = config["device"]

EMBEDDING_MODEL_PARAMS: Dict[Any, Any] = {}

BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
严格按照如下规则输出：
1. 如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。
2. 如果提问不符合逻辑，请回答无法回答。
3. 如果问题可以从资料中获得，则请逐步回答。
4. 单独输出引用原文部分，将回答中用到的资料内容原文复制过来

资料：
{#RELATED_DOCUMENT#}

问题：{#QUESTION#}

输出格式示例：
【问题解答】
XXX（这里写你的回答内容）

【引用原文】
- 原文片段1：XXX（资料中对应的内容）
- 原文片段2：XXX（资料中对应的内容）
'''


def load_embdding_model(model_name: str, model_path: str) -> None:
    """
    加载编码模型（将文本转为向量）
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    # sbert模型（中文常用嵌入模型，轻量且效果好）
    if model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        # 使用SentenceTransformer库加载模型，并存到全局字典
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(model_path)


# 加载重排序模型
def load_rerank_model(model_name: str, model_path: str) -> None:
    """
    加载重排序模型（对检索结果打分排序）
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-reranker-base"]:
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(device)


if config["rag"]["use_embedding"]:
    # 获取当前使用的嵌入模型名称
    model_name = config["rag"]["embedding_model"]
    # 获取模型本地路径
    model_path = config["models"]["embedding_model"][model_name]["local_url"]

    print(f"Loading embedding model {model_name} from model_path...")
    load_embdding_model(model_name, model_path)

if config["rag"]["use_rerank"]:
    model_name = config["rag"]["rerank_model"]
    model_path = config["models"]["rerank_model"][model_name]["local_url"]

    print(f"Loading rerank model {model_name} from model_path...")
    load_rerank_model(model_name, model_path)


"""
RAG类，整合了文档提取、向量生成、检索融合、LLM 交互所有功能
"""


class RAG:
    def __init__(self):
        # md 文档路径
        self.md_path = config["rag"]["md_path"]
        # 从配置中获取当前使用的嵌入模型和重排序模型名称
        self.embedding_model = config["rag"]["embedding_model"]
        self.rerank_model = config["rag"]["rerank_model"]

        # 是否启用重排序（从配置读取）
        self.use_rerank = config["rag"]["use_rerank"]

        # 嵌入向量的维度
        self.embedding_dims = config["models"]["embedding_model"][
            config["rag"]["embedding_model"]
        ]["dims"]

        # 分块参数（从配置读取）
        self.chunk_size = config["rag"]["chunk_size"]  # 块大小
        self.chunk_overlap = config["rag"]["chunk_overlap"]  # 重叠大小
        self.chunk_candidate = config["rag"]["chunk_candidate"]  # 检索后保留的候选块数量

        # 初始化LLM客户端（连接大模型，支持OpenAI API格式的模型）
        self.client = OpenAI(
            api_key=config["rag"]["llm_api_key"],  # API密钥
            base_url=config["rag"]["llm_base"]  # API地址
        )
        self.llm_model = config["rag"]["llm_model"]  # 使用的LLM模型

        try:
            # 先检查ES里是否已有该文档的chunk（避免重复入库）
            es_check = es.search(
                index="chunk_info",
                body={"query": {"match": {"document_id": "rag_demo_001"}}}
            )
            if es_check["hits"]["total"]["value"] == 0:
                # 没有则执行入库
                self.extract_content(
                    document_id="rag_demo_001",
                    title="RAG自动开发手册",
                    file_path=self.md_path
                )
                print("文档首次入库成功！")
            else:
                print("文档已存在，跳过入库～")
        except Exception as e:
            print(f"入库检查/执行失败：{str(e)}")

    # md内容提取
    def _extract_md_content(self, document_id, title, md_path) -> bool:
        # 提取md文件内容，并存储到ES
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            headers_to_split_on = [
                ("#", "一级标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            docs = markdown_splitter.split_text(md_content)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,    # 每块最大300字
                chunk_overlap=self.chunk_overlap,  # 相邻块重叠50字
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "，"]  # 按中文标点优先拆分
            )
            chunks = text_splitter.split_documents(docs)
            chunk_texts = [chunk.page_content for chunk in chunks]
            # 批量生成所有分块的嵌入向量
            embedding_vector = self.get_embedding(chunk_texts)
            # 遍历每个分块，存储到ES
            for chunk_idx in range(1, len(chunks) + 1):
                chunk_data = {
                    "document_id": document_id,
                    "chunk_id": chunk_idx,  
                    "chunk_content": chunks[chunk_idx - 1].page_content, 
                    "chunk_images": [],
                    "chunk_tables": [],
                    "embedding_vector": embedding_vector[chunk_idx - 1], 
                    "metadata": chunks[chunk_idx - 1].metadata 
                }
                es.index(index="chunk_info", document=chunk_data)

            # 存储文档元数据到document_meta索引
            document_data = {
                "document_id": document_id,
                "document_name": title,  # 文档标题
                "file_path": md_path,  # 文档本地路径
            }
            es.index(index="document_meta", document=document_data)
            return True
        except Exception as e:
            print("打开文件失败")
            return False

    # 统一内容提取入口
    def extract_content(self, document_id, title, file_path):
        # 根据文件类型调用对应的提取方法
        self._extract_md_content(document_id, title, file_path)

        print("提取完成", document_id, file_path)

    # 生成文本嵌入向量
    def get_embedding(self, text) -> np.ndarray:
        """
        对文本进行编码，生成嵌入向量
        :param text: 待编码文本
        :return: 编码结果
        """
        # 如果使用BGE系列嵌入模型
        if self.embedding_model in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
            # 调用模型编码，normalize_embeddings=True表示向量归一化（方便计算余弦相似度）
            return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)

        # 如果使用其他模型，这里会抛出“未实现”的异常
        raise NotImplemented

    # 重排序打分
    def get_rank(self, text_pair) -> np.ndarray:
        """
        对查询-文档进行相关性打分（重排序）
        :param text_pair: 待排序文本 文本对列表，每个元素是[查询文本, 候选文档文本]
        :return: 匹配打分结果
        """
        if self.rerank_model in ["bge-reranker-base"]:
            with torch.no_grad():
                # 对文本对进行分词（转换为模型可处理的格式）
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair,
                    padding=True,  # 短文本补全到最长文本长度
                    truncation=True,  # 长文本截断到模型最大长度（如512）
                    return_tensors='pt',  # 返回PyTorch张量
                    max_length=512,  # 最大长度限制
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                # 模型推理，获取打分结果
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores

        raise NotImplemented

    # 文档检索
    def query_document(self, query: str) -> List[Dict]:
        # 1. 全文检索，指定一个知识库检索，bm25打分
        # 向ES的"chunk_info"索引发送查询，指定“知识库ID”和“关键词匹配”
        word_search_response = es.search(
            index="chunk_info",
            body={
                "query": {
                    "bool": {  # 组合条件查询
                        "must": [  # 必须满足：chunk_content包含query的关键词
                            {
                                "match": {
                                    "chunk_content": query
                                }
                            }
                        ],
                    }
                },
                "size": 50  # 返回前50条结果
            },
            # 只返回需要的字段（减少数据传输）
            fields=["chunk_id", "document_id", "chunk_content"],
            source=False,  # 不返回原始文档，只返回指定fields
        )

        # 2. 语义检索（kNN算法）：基于向量相似度，找和查询向量最像的文档块
        # 先把用户查询转成向量
        embedding_vector = self.get_embedding(query)  # 查询文本的嵌入向量 编码
        # 构造kNN检索参数
        knn_query = {
            "field": "embedding_vector",  # 检索的向量字段
            "query_vector": embedding_vector,  # 查询向量
            "k": 50,  # 返回最相似的50条
            "num_candidates": 100,  # 初步计算得到top100的待选文档，筛选最相关的50个
        }
        # 发送kNN检索请求
        vector_search_response = es.search(
            index="chunk_info",
            knn=knn_query,  # 使用kNN检索
            fields=["chunk_id", "document_id", "chunk_content"],
            source=False,
        )

        # 3. RRF融合（Reciprocal Rank Fusion）：融合两种检索结果，提升准确性
        # RRF原理：每个检索结果的“排名”越靠前，贡献的分数越高，公式是 1/(排名 + k)
        k = 60  # RRF参数（控制排名权重）
        fusion_score = {}  # 存储每个文档块的融合分数
        search_id2record = {}  # 存储文档块ID到内容的映射

        # 处理全文检索结果，计算RRF分数
        # 默认是按照 _score（相关性分数）从高到低排序的
        for idx, record in enumerate(word_search_response['hits']['hits']):
            _id = record["_id"]  # 文档块在ES中的唯一ID
            fusion_score[_id] = fusion_score.get(_id, 0.0) + 1 / (idx + k)
            # RRF分数公式：1/(排名+ k)，排名越靠前（idx越小），分数越高
            if _id not in search_id2record:
                search_id2record[_id] = {
                    "chunk_id": record["fields"]["chunk_id"][0] if "chunk_id" in record["fields"] else "",
                    "document_id": record["fields"]["document_id"][0] if "document_id" in record["fields"] else "",
                    "chunk_content": record["fields"]["chunk_content"][0] if "chunk_content" in record["fields"] else ""
                }

        # 处理语义检索结果，计算RRF分数（逻辑同上）
        for idx, record in enumerate(vector_search_response['hits']['hits']):
            _id = record["_id"]
            fusion_score[_id] = fusion_score.get(_id, 0.0) + 1 / (idx + k)
            if _id not in search_id2record:
                search_id2record[_id] = {
                    "chunk_id": record["fields"]["chunk_id"][0] if "chunk_id" in record["fields"] else "",
                    "document_id": record["fields"]["document_id"][0] if "document_id" in record["fields"] else "",
                    "chunk_content": record["fields"]["chunk_content"][0] if "chunk_content" in record["fields"] else ""
                }

        # 4. 按融合分数排序，取前N个候选（chunk_candidate）
        # 对融合分数从高到低排序（sorted默认升序，reverse=True改为降序）
        sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
        # 取前chunk_candidate个结果
        sorted_records = [search_id2record[x[0]] for x in sorted_dict if x[0] in search_id2record][:self.chunk_candidate]
        sorted_content = [x["chunk_content"] for x in sorted_records]

        # 5. 如果启用重排序，对候选结果再次打分排序
        if self.use_rerank:
            # 构造文本对：[查询, 候选文档块内容]
            text_pair = []
            for record, content in zip(sorted_records, sorted_content):
                if content and len(content) > 10:
                    text_pair.append([query, content])

            if text_pair:
                rerank_score = self.get_rank(text_pair)  # 重排序打分
                # 按分数从高到低排序，获取索引
                rerank_idx = np.argsort(rerank_score)[::-1][:5]

                sorted_records = [sorted_records[x] for x in rerank_idx]
                sorted_content = [sorted_content[x] for x in rerank_idx]

        return sorted_records

    # RAG 聊天
    def chat_with_rag(
            self,
            messages: List[Message],  # 聊天消息列表（包含用户问题）
    ):
        # 1. 单轮对话（只有用户的问题，没有历史上下文）
        if len(messages) == 1:
            query = messages[0].content  # 提取用户问题
            related_records = self.query_document(query)  # 检索到相关的文档
            print(related_records)
            # 提取文档块内容，拼接为字符串
            # 这里的[0]，就是为了从列表中提取出那个唯一的字符串元素，从[...]变为...
            related_document = '\n'.join([x["chunk_content"] for x in related_records])

            # 填充提示词模板（替换时间、资料、问题）
            rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
                .replace("{#QUESTION#}", query) \
                .replace("{#RELATED_DOCUMENT#}", related_document)

            # 调用LLM生成回答（基于资料）
            rag_response = self.chat(
                [{"role": "user", "content": rag_query}],
                0.7,  # top_p参数（控制输出多样性：0.9表示保留90%概率的词）
                0.9  # temperature参数（控制随机性：0.7适中，越高越随机）
            ).content
            # 把LLM的回答添加到消息列表（角色为"system"，表示系统回答）
            messages.append({"role": "system", "content": rag_response})
        # 2. 多轮对话（暂不结合知识库，直接调用LLM基于上下文回答）
        else:
            # 调用LLM，基于历史消息生成回答
            normal_response = self.chat(
                messages,  # 包含历史上下文（用户问题+之前的回答）
                0.7, 0.9
            ).content
            messages.append({"role": "system", "content": normal_response})

        # messages.append({"role": "system", "content": rag_response})
        # 返回包含“用户问题+系统回答”的消息列表
        return messages

    # 调用LLM
    def chat(self, messages: List[Message], top_p: float, temperature: float) -> Any:
        # 调用OpenAI格式的LLM API，生成回答
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages, 
            top_p=top_p, 
            temperature=temperature 
        )
        return completion.choices[0].message


