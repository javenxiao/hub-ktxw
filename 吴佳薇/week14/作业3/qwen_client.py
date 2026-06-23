import json
import re
from typing import Dict, Any, Optional
import httpx
from dataclasses import dataclass
from enum import Enum
from tool_registry import find_most_similar_tool

class EnhancedQwenClientWithMCP:
    def __init__(self):
        self.llm_client = httpx.Client(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30,
            headers={
                "Authorization": f"Bearer {"API-KEY"}",
                "Content-Type": "application/json"
            }
        )
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询的完整流程

        Returns:
            包含处理结果的字典
        """
        result = {
            'success': False,
            'tool_name': None,
            'similarity': 0,
            'extracted_params': {},
            'result': None,
            'answer': None,
            'error': None
        }

        try:
            # 1. 使用RAG找到最匹配的工具
            matched_tool, similarity = find_most_similar_tool(query)

            if not matched_tool or similarity < 0.3:
                result['error'] = "没有找到适合处理您问题的工具。"
                return result

            result['tool_name'] = matched_tool['name']
            result['similarity'] = similarity

            print(f"🎯 匹配工具: {matched_tool['name']} (相似度: {similarity:.3f})")

            # 2. 使用LLM提取参数
            extracted_params = self.extract_parameters(query, matched_tool)
            result['extracted_params'] = extracted_params

            # 3. 调用工具计算
            try:
                func_result = matched_tool['function'](**extracted_params)
                result['result'] = func_result
                result['success'] = True

                # 5. 使用LLM生成自然语言回答
                answer = self._generate_answer_with_llm(query, matched_tool, extracted_params, func_result)
                result['answer'] = answer

            except Exception as e:
                result['error'] = f"计算错误: {str(e)}"

        except Exception as e:
            result['error'] = f"处理查询时出错: {str(e)}"

        return result

    def extract_parameters(self, query: str, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用Qwen LLM从用户查询中提取参数

        Args:
            query: 用户查询文本
            tool_schema: 工具的参数schema

        Returns:
            提取的参数字典
        """
        # 构建系统提示词
        system_prompt = """你是一个参数提取专家。你的任务是从用户查询中提取指定工具所需的参数。

        规则：
        1. 只提取工具所需的参数，不要添加额外的参数
        2. 如果用户查询中没有明确给出参数值，请设为null
        3. 确保参数类型正确（数字、字符串、布尔值等）
        4. 返回格式必须是有效的JSON
        5. 对于数值参数，确保提取的是数字而不是文本描述

        示例：
        用户查询："计算x=3时的二次方程值"
        工具参数：{"x": "float"}
        输出：{"x": 3}
        """

        # 构建用户提示词
        user_prompt = f"""请从以下用户查询中提取工具参数：

        用户查询：{query}

        工具信息：
        - 工具名称：{tool_schema.get('name', '未知工具')}
        - 工具描述：{tool_schema.get('description', '无描述')}
        - 参数定义：{json.dumps(tool_schema.get('parameters', {}), ensure_ascii=False, indent=2)}
        
        请以JSON格式返回提取的参数，格式如：{{"参数名": 值, ...}}
        如果无法确定参数值，设为null。
        
        直接返回JSON，不要添加额外解释。"""

        try:
            # 调用Qwen API
            response = self.llm_client.post(
                "/chat/completions",
                json={
                    "model": "qwen-max",
                    "messages": [{"role":"system","content":system_prompt},{"role":"user","content": user_prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,  # 低温度确保一致性
                    "max_tokens": 1000
                }
            )

            response.raise_for_status()
            result = response.json()

            # 解析响应
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")

            # 清理和解析JSON
            content = content.strip()

            # 移除可能的markdown代码块
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # 解析JSON
            extracted_params = json.loads(content)

            # 转换null为None
            for key, value in extracted_params.items():
                if value == "null" or value is None:
                    extracted_params[key] = None

            return extracted_params

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {content[:200]}...")
            return {}
        except httpx.HTTPError as e:
            print(f"HTTP请求错误: {e}")
            return {}
        except Exception as e:
            print(f"提取参数时出错: {e}")
            return {}

    def _generate_answer_with_llm(self, query: str, tool_info: Dict[str, Any],
                                  params: Dict[str, Any], result: Any) -> str:
        """使用LLM生成自然语言回答"""
        # 构建提示词
        system_prompt = """你是一个专业的技术助手。根据用户查询和计算结果，生成一个自然、专业的回答。

        回答要求：
        1. 简洁明了，用中文回答
        2. 包含计算结果和参数信息
        3. 解释结果的意义
        4. 如果适用，提供进一步建议
        5. 保持友好和专业
        """

        user_prompt = f"""请根据以下信息生成回答：

        用户查询：{query}
        
        使用的工具：{tool_info.get('name', '未知工具')}
        工具描述：{tool_info.get('description', '')}
        
        输入参数：
        {json.dumps(params, ensure_ascii=False, indent=2)}
        
        计算结果：{result}
        
        请生成一个自然的回答，解释这个计算过程和结果意义。"""

        try:
            # 调用Qwen API生成回答
            response = self.llm_client.client.post(
                "/chat/completions",
                json={
                    "model": "qwen-max",
                    "messages": [{"role":"system","content":system_prompt},{"role":"user","content": user_prompt}],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )

            response.raise_for_status()
            result_data = response.json()

            content = result_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return content.strip()

        except Exception as e:
            print(f"生成回答时出错: {e}")
            # 返回简单回答
            return f"计算结果为: {result}\n参数: {params}"
