from typing import Optional, List, Union, Any, Dict
import math
from collections import defaultdict


class SemanticRouter:
    def __init__(
            self
    ):
        # 存储路由规则：{目标: [示例问题列表]}
        self.routes = {}
        # 存储词汇表用于计算相似度
        self.vocabulary = set()

    def add_route(self, questions: List[str], target: str):
        """
        添加路由规则
        questions: 示例问题列表
        target: 目标类别
        """
        # 存储路由
        self.routes[target] = questions

        # 更新词汇表（简单分词：按空格分割）
        for question in questions:
            words = question.lower().split()
            self.vocabulary.update(words)

        print(f"已添加路由 '{target}'，包含 {len(questions)} 个示例问题")
        print(f"词汇表大小: {len(self.vocabulary)}")

    def _text_to_vector(self, text: str):
        """将文本转换为向量（简单的词频向量）"""
        # 将文本转换为小写并分词
        words = text.lower().split()

        # 创建词频向量
        vector = defaultdict(int)
        for word in words:
            vector[word] += 1

        return vector

    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        # 获取所有独特的词语
        all_words = set(vec1.keys()) | set(vec2.keys())

        # 计算点积
        dot_product = 0
        for word in all_words:
            dot_product += vec1.get(word, 0) * vec2.get(word, 0)

        # 计算向量长度
        magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

        # 避免除以零
        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def route(self, question: str, threshold: float = 0.3):
        """
        路由一个问题
        question: 输入问题
        threshold: 相似度阈值（低于此值认为不匹配）
        """
        if not question:
            return None

        # 将输入问题转换为向量
        question_vector = self._text_to_vector(question)

        best_match = None
        best_similarity = 0

        # 与每个路由的示例问题比较
        for target, examples in self.routes.items():
            for example in examples:
                # 将示例问题转换为向量
                example_vector = self._text_to_vector(example)

                # 计算相似度
                similarity = self._cosine_similarity(question_vector, example_vector)

                # 更新最佳匹配
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = target

        # 如果相似度达到阈值，返回最佳匹配
        if best_match and best_similarity >= threshold:
            print(f"匹配: '{question}' -> {best_match} (相似度: {best_similarity:.2f})")
            return best_match
        else:
            print(f"未匹配: '{question}' (最高相似度: {best_similarity:.2f})")
            return None

    def __call__(self, question: str, threshold: float = 0.3):
        """使路由器可调用"""
        return self.route(question, threshold)


if __name__ == "__main__":
    # 创建路由器
    router = SemanticRouter()

    # 添加路由规则
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "Hello there", "Hey friend"],
        target="greeting"
    )

    router.add_route(
        questions=["如何退货", "我要退货", "退货流程", "怎么退货"],
        target="refund"
    )

    # 测试路由功能
    print("\n=== 测试路由 ===")

    # 测试完全匹配
    print("1. 完全匹配测试:")
    router("Hi, good morning")
    router("如何退货")

    # 测试部分匹配
    print("\n2. 部分匹配测试:")
    router("Hi, good morning everyone")
    router("早上好")
    router("退货怎么弄")

    # 测试不匹配的情况
    print("\n3. 不匹配测试:")
    router("今天天气怎么样")
    router("帮我订个酒店")
