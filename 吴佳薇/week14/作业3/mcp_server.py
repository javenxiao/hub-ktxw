import os
from dotenv import load_dotenv
from qwen_client import EnhancedQwenClientWithMCP


def main():
    # 创建客户端
    client = EnhancedQwenClientWithMCP("sk-6511723ce36a49f21a67e69b6ace52ab")

    # 测试查询
    queries = [
        "计算静息心率，年龄50，每日步数10千步",
        "传播项目的内容质量是0.85，有4个渠道，参与度0.75，持续10天",
        "当前牛群数量是150头，增长率12%，环境容量1200头",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        result = client.process_query(query)

        if result['success']:
            print(f"工具: {result['tool_name']}")
            print(f"参数: {result['extracted_params']}")
            print(f"结果: {result['result']}")
            print(f"回答: {result['answer'][:100]}...")
        else:
            print(f"X错误: {result['error']}")


if __name__ == "__main__":
    main()
