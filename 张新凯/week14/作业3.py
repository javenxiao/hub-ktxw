import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, AsyncIterator

import jieba
import numpy as np
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, ModelSettings
from agents.mcp import ToolFilterStatic
from agents.mcp.server import MCPServerSse
from fastmcp import Client
from openai.types.responses import ResponseOutputItemDoneEvent, ResponseFunctionToolCall, ResponseTextDeltaEvent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MCPConfig:
    """MCP服务器配置"""
    url: str = "http://localhost:8900/sse"
    name: str = "SSE Python Server"
    client_timeout: int = 20
    cache_tools: bool = False


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "qwen-max"
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    parallel_tool_calls: bool = False


class ToolSelector:
    """工具选择器，封装相似度计算逻辑"""

    def __init__(self, vectorizer: Optional[TfidfVectorizer] = None):
        self.vectorizer = vectorizer or TfidfVectorizer()
        self._tool_cache: Dict[str, Tuple[List[str], List[str]]] = {}

    async def get_tools_from_server(self, mcp_url: str) -> Tuple[List[str], List[str]]:
        """从MCP服务器获取工具列表"""
        cache_key = mcp_url
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        try:
            async with Client(mcp_url) as client:
                tools = await client.list_tools()
                names = [tool.name for tool in tools]
                descriptions = [tool.description for tool in tools]
                self._tool_cache[cache_key] = (names, descriptions)
                return names, descriptions
        except Exception as e:
            logger.error(f"Failed to fetch tools from {mcp_url}: {e}")
            raise

    def preprocess_chinese_text(self, texts: List[str]) -> List[str]:
        """预处理中文文本，进行分词"""
        return [" ".join(jieba.lcut(text)) for text in texts]

    def select_relevant_tools(
            self,
            user_query: str,
            tool_names: List[str],
            tool_descriptions: List[str],
            top_k: int = 5
    ) -> List[str]:
        """基于TF-IDF和余弦相似度选择相关工具"""
        if not tool_descriptions:
            return []

        # 预处理文本
        processed_descriptions = self.preprocess_chinese_text(tool_descriptions)
        processed_query = self.preprocess_chinese_text([user_query])

        # 训练TF-IDF向量器
        description_vectors = self.vectorizer.fit_transform(processed_descriptions)
        query_vector = self.vectorizer.transform(processed_query)

        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, description_vectors)[0]

        # 选择top_k个最相关的工具
        top_indices = np.argsort(similarities)[::-1][:top_k]
        selected_tools = [tool_names[i] for i in top_indices if similarities[i] > 0]

        # 记录选择结果
        logger.info(f"Selected {len(selected_tools)} tools: {selected_tools}")
        for idx in top_indices[:3]:
            logger.debug(f"Tool: {tool_names[idx]}, Similarity: {similarities[idx]:.3f}")

        return selected_tools


class MCPAgentRunner:
    """MCP Agent运行器"""

    def __init__(
            self,
            mcp_config: MCPConfig,
            model_config: ModelConfig
    ):
        self.mcp_config = mcp_config
        self.model_config = model_config
        self.tool_selector = ToolSelector()

        # 配置OpenAI客户端
        self._setup_openai_client()

    def _setup_openai_client(self) -> None:
        """配置OpenAI客户端"""
        api_key = self.model_config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.model_config.base_url,
        )

    async def create_mcp_server(self, allowed_tools: List[str]) -> MCPServerSse:
        """创建MCP服务器实例"""
        tool_filter = ToolFilterStatic(allowed_tool_names=allowed_tools)

        return MCPServerSse(
            name=self.mcp_config.name,
            params={"url": self.mcp_config.url},
            cache_tools_list=self.mcp_config.cache_tools,
            tool_filter=tool_filter,
            client_session_timeout_seconds=self.mcp_config.client_timeout,
        )

    def create_agent(self, mcp_server: MCPServerSse) -> Agent:
        """创建Agent实例"""
        return Agent(
            name="Tool Agent",
            instructions="""请遵循以下步骤：
            1. 首先复述用户问题，确认理解正确
            2. 说明你将调用的工具及其用途
            3. 调用适当的工具并展示结果
            4. 根据结果提供总结和建议

            如果遇到问题，请诚实地说明情况并给出替代方案。""",
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=self.model_config.model_name,
                openai_client=self.openai_client,
            ),
            model_settings=ModelSettings(
                parallel_tool_calls=self.model_config.parallel_tool_calls
            )
        )

    async def run_with_tool_selection(
            self,
            user_query: str,
            top_k: int = 5
    ) -> AsyncIterator[Dict[str, Any]]:
        """运行带工具选择的Agent"""
        try:
            # 1. 获取所有可用工具
            logger.info("Fetching available tools from MCP server...")
            tool_names, tool_descriptions = await self.tool_selector.get_tools_from_server(
                self.mcp_config.url
            )

            if not tool_names:
                yield {"type": "error", "message": "No tools available from MCP server"}
                return

            # 2. 选择相关工具
            logger.info("Selecting relevant tools based on user query...")
            selected_tools = self.tool_selector.select_relevant_tools(
                user_query, tool_names, tool_descriptions, top_k
            )

            if not selected_tools:
                yield {"type": "error", "message": "No relevant tools found for the query"}
                return

            # 3. 创建MCP服务器和Agent
            async with await self.create_mcp_server(selected_tools) as mcp_server:
                agent = self.create_agent(mcp_server)

                # 4. 运行Agent并流式输出
                logger.info("Starting Agent execution...")
                result = Runner.run_streamed(
                    agent,
                    input=user_query,
                    run_config=RunConfig(
                        model_settings=ModelSettings(
                            parallel_tool_calls=self.model_config.parallel_tool_calls
                        )
                    )
                )

                async for event in result.stream_events():
                    if event.type == "raw_response_event" and hasattr(event, 'data'):
                        if isinstance(event.data, ResponseOutputItemDoneEvent):
                            if isinstance(event.data.item, ResponseFunctionToolCall):
                                tool_call = event.data.item
                                yield {
                                    "type": "tool_call",
                                    "tool_name": tool_call.name,
                                    "arguments": tool_call.arguments
                                }

                        elif isinstance(event.data, ResponseTextDeltaEvent):
                            yield {
                                "type": "text_delta",
                                "content": event.data.delta
                            }

                    elif event.type == "run_item_stream_event":
                        if hasattr(event, 'name') and event.name == "tool_output":
                            yield {
                                "type": "tool_output",
                                "data": event
                            }

        except asyncio.TimeoutError:
            yield {"type": "error", "message": "Request timeout"}
            logger.error("Agent execution timeout")
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            logger.exception("Agent execution failed")


class ConfigurationManager:
    """配置管理器"""

    @staticmethod
    def load_from_env() -> Tuple[MCPConfig, ModelConfig]:
        """从环境变量加载配置"""
        return (
            MCPConfig(
                url=os.getenv("MCP_SERVER_URL", "http://localhost:8900/sse"),
                name=os.getenv("MCP_SERVER_NAME", "SSE Python Server"),
                client_timeout=int(os.getenv("MCP_CLIENT_TIMEOUT", "20")),
                cache_tools=os.getenv("MCP_CACHE_TOOLS", "false").lower() == "true"
            ),
            ModelConfig(
                model_name=os.getenv("MODEL_NAME", "qwen-max"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL",
                                   "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                parallel_tool_calls=os.getenv("PARALLEL_TOOL_CALLS", "false").lower() == "true"
            )
        )


async def main():
    """主函数"""
    # 加载配置
    mcp_config, model_config = ConfigurationManager.load_from_env()

    # 创建Agent运行器
    runner = MCPAgentRunner(
        mcp_config=mcp_config,
        model_config=model_config
    )

    # 用户查询
    user_query = "已知一个人的体重是70kg，请问他的基础代谢率大概是多少？"

    # 运行Agent
    print(f"用户查询: {user_query}\n")
    print("=" * 50)

    try:
        async for event in runner.run_with_tool_selection(user_query, top_k=5):
            if event["type"] == "text_delta":
                print(event["content"], end="", flush=True)
            elif event["type"] == "tool_call":
                print(f"\n[调用工具] {event['tool_name']}: {event['arguments']}")
            elif event["type"] == "tool_output":
                print(f"\n[工具输出] 收到响应")
            elif event["type"] == "error":
                print(f"\n[错误] {event['message']}")
                break

        print("\n" + "=" * 50)
        print("执行完成")

    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"\n执行失败: {e}")


if __name__ == "__main__":
    # 设置环境变量（在实际使用中应该通过.env文件或环境配置）
    os.environ.setdefault("OPENAI_API_KEY", "sk-xxx")
    os.environ.setdefault("OPENAI_BASE_URL",
                          "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 禁用agents的跟踪（可选）
    from agents import set_tracing_disabled

    set_tracing_disabled(True)

    # 运行主程序
    asyncio.run(main())
