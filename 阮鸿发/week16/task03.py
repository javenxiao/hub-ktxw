
from semantic_router.encoders import OpenAIEncoder
from semantic_router.routers import SemanticRouter as RouteLayer

from semantic_router import Route


# 1. 指定本地 Ollama 服务地址
LOCAL_URL = "http://localhost:11434/v1"

# 2. 创建 Encoder，指向本地
encoder = OpenAIEncoder(
    name="qwen3:0.6b",  # 指定模型名称
    base_url=LOCAL_URL,  # 关键：指向本地
    api_key=""  # Ollama 通常不需要 key
)

# 3. 定义意图 (Route)...
politics_route = Route(
    name="politics",  # 意图名称
    utterances=[  # 典型的用户问法示例
        "你怎么看待现在的政治局势？",
        "你支持哪个政党？",
        "总统最近的政策怎么样？",
        "这届政府要毁了这个国家！",
        "他们能拯救国家的经济！"
    ]
)

# 定义“闲聊”意图
chitchat_route = Route(
    name="chitchat",
    utterances=[
        "今天天气怎么样？",
        "最近好吗？",
        "今天天气真好",
        "这鬼天气真是糟糕",
        "我们去喝一杯吧"
    ]
)

# 定义“技术支持”意图
tech_support_route = Route(
    name="tech_support",
    utterances=[
        "我的电脑蓝屏了怎么办？",
        "软件打不开了",
        "怎么重置密码？",
        "网络连接不上"
    ]
)

# 将所有意图汇总到一个列表中
routes = [politics_route, chitchat_route, tech_support_route]

# 4. 创建路由层
# 这里的计算现在都会在本地 Ollama 中进行
route_layer = RouteLayer(encoder=encoder, routes=routes)

# 5. 测试
result = route_layer("我的电脑蓝屏了")
print(result.name)  # 应该输出 tech_support







