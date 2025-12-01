"""
semantic_parser_prompt.py
语义解析助手模块：将自然语言转换为结构化语义数据。
"""

import json
from openai import OpenAI


class SemanticParser:
    def __init__(self, api_key: str, domain_file='domains.txt', intent_file='intents.txt', slots_file='slots.txt'):
        self.client = OpenAI(api_key=api_key,
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.domains = self._load_list(domain_file)
        self.intents = self._load_list(intent_file)
        self.slots = self._load_list(slots_file)

    def _load_list(self, file_path):
        """读取业务配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"⚠️ 文件未找到: {file_path}")
            return []

    def _build_prompt(self, user_input: str):
        """构造 LLM 提示词"""
        return f"""
你是一个精准的语义解析助手，负责将自然语言转换为结构化数据。

【任务说明】
请根据以下业务定义，从用户输入中识别出意图(intent)、领域(domain)和槽位(slots)。

- 可能的 intent 取值：{self.intents}
- 可能的 domain 取值：{self.domains}
- 可能的 slots 键名：{self.slots}

【输出格式】
请严格输出 JSON，格式如下：
{{
  "text": "{user_input}"
  "intent": "",
  "domain": "",
  "slots": {{}}
}}

不要输出任何解释性文字。

【示例】
用户输入：查询许昌到中山的汽车。
输出：
{{
  "text": "查询许昌到中山的汽车"
  "intent": "QUERY",
  "domain": "bus",
  "slots": {{
    "Src": "许昌",
    "Dest": "中山"
  }}
}}

【现在开始】
用户输入：{user_input}
"""

    def parse(self, text: str):
        """执行语义解析"""
        prompt = self._build_prompt(text)
        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个高精度的语义解析模型。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        result = response.choices[0].message.content.strip()

        # 尝试将结果转为 JSON 对象
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            print("⚠️ 模型输出非标准 JSON，将原文返回。")
            parsed = {"raw_output": result}

        return parsed


if __name__ == "__main__":
    # 示例运行
    OPENAI_API_KEY = "sk-******"
    parser = SemanticParser(api_key=OPENAI_API_KEY)
    text = "茅屋为秋风所破歌杜甫"
    result = parser.parse(text)

    print("\n🧠 输入：", text)
    print("📦 输出：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
