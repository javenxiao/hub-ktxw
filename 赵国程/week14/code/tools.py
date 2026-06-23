import asyncio
import requests  # type: ignore
import math
from fastmcp import FastMCP, Client

mcp = FastMCP(
    name="Math-MCP-Server",
    instructions="""This server contains some api of math.""",
)


@mcp.tool
def get_bmr_rate(weight, height):
    """
    计算基础代谢率（BMR），单位：千卡/天。
    公式：BMR = 10 * weight + 6.25 * height - 5 * age + 5 （简化版，未含年龄）
    此处采用文档中的简化形式：BMR = weight * 2 + height / 100 * 100
    实际按文档描述：BMR = weight * 2 + height（但单位需一致）
    根据常见理解，此处实现为：
        BMR ≈ 10 * weight(kg) + 6.25 * height(cm)
    但原文公式较模糊，按字面 "weight*2 + height" 实现。

    参数:
        weight (float): 体重，单位 kg
        height (float): 身高，单位 cm

    返回:
        float: 基础代谢率（BMR）
    """
    return weight * 2 + height


@mcp.tool
def simulate_linear_dynamic_system(y_prev, x1, x2, x3, x4, x5, a, b, c, d, e):
    """
    一阶线性差分方程模型：y(t) = a*x1 + b*x2 + c*x3 + d*x4 + e*x5 + y_prev

    参数:
        y_prev (float): 上一时刻系统状态
        x1~x5 (float): 当前输入变量
        a~e (float): 对应系数

    返回:
        float: 当前时刻系统输出 y(t)
    """
    return a * x1 + b * x2 + c * x3 + d * x4 + e * x5 + y_prev


@mcp.tool
def predict_cattle_population(Nt, r, K):
    """
    牛群数量逻辑斯蒂增长模型：N(t+1) = Nt + r * Nt * (1 - Nt / K)

    参数:
        Nt (float): 当前牛群数量
        r (float): 内禀增长率
        K (float): 环境承载力（最大容量）

    返回:
        float: 下一时刻的牛群数量 N(t+1)
    """
    return Nt + r * Nt * (1 - Nt / K)


@mcp.tool
def calculate_influence(content_quality, channels, engagement, time):
    """
    文化传播综合影响力模型：
    Influence = (content_quality * channels * engagement) / (1 + time)

    参数:
        content_quality (float): 内容质量（0~1）
        channels (int): 传播渠道数量
        engagement (float): 用户参与度（0~1）
        time (float): 时间（单位：月或年）

    返回:
        float: 综合影响力指数
    """
    return (content_quality * channels * engagement) / (1 + time)


@mcp.tool
def evaluate_quadratic_model(x, a=1, b=0, c=0):
    """
    二次函数确定性模型：y = a*x^2 + b*x + c

    参数:
        x (float): 自变量
        a, b, c (float): 二次项、一次项、常数项系数（默认 y = x^2）

    返回:
        float: 函数值 y
    """
    return a * x ** 2 + b * x + c


@mcp.tool
def predict_system_state(x1, x2, x3, y_t1, y_t2, a, b, c, d):
    """
    三输入变量二阶差分方程模型：
    y(t) = a*x1 + b*x2 + c*x3 + d*y(t-1) + (1-d)*y(t-2)

    参数:
        x1, x2, x3 (float): 当前输入
        y_t1 (float): t-1 时刻输出
        y_t2 (float): t-2 时刻输出
        a, b, c, d (float): 模型系数

    返回:
        float: 当前时刻系统状态 y(t)
    """
    return a * x1 + b * x2 + c * x3 + d * y_t1 + (1 - d) * y_t2


@mcp.tool
def predict_learning_score(x1, x2, x3, x4, w1=1, w2=1, w3=1, w4=1, alpha=1, beta=0):
    """
    教育培训学习效果Sigmoid模型：
    score = 1 / (1 + exp(-alpha * (w1*x1 + w2*x2 + w3*x3 + w4*x4) + beta))

    参数:
        x1~x4 (float): 学习因素（如课时、练习量等）
        w1~w4 (float): 权重
        alpha (float): 陡峭度参数
        beta (float): 偏移量

    返回:
        float: 学习效果评分（0~1）
    """
    linear_comb = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4
    return 1 / (1 + math.exp(-alpha * linear_comb + beta))


@mcp.tool
def moisture_content(M0, k, t):
    """
    食品干燥水分含量模型（指数衰减）：
    M(t) = M0 * exp(-k * t)

    参数:
        M0 (float): 初始水分含量
        k (float): 干燥速率常数
        t (float): 时间

    返回:
        float: t 时刻的水分含量
    """
    return M0 * math.exp(-k * t)


@mcp.tool
def predict_crop_yield(F, I, T, a=0.5, b=0.3, c=0.2):
    """
    农业作物产量预测模型（线性加权）：
    Yield = a*F + b*I + c*T
    其中 F=肥料, I=灌溉, T=温度（已标准化）

    参数:
        F, I, T (float): 肥料、灌溉、温度（建议归一化到 0~1）
        a, b, c (float): 权重系数

    返回:
        float: 预测产量（相对值）
    """
    return a * F + b * I + c * T


@mcp.tool
def predict_daily_orders(ad_spend, discount_rate, prev_orders, alpha=0.4, beta=0.3, gamma=0.3):
    """
    电商每日订单预测模型：
    orders = alpha * ad_spend + beta * discount_rate + gamma * prev_orders

    参数:
        ad_spend (float): 广告投入（标准化）
        discount_rate (float): 折扣力度（0~1）
        prev_orders (float): 前一日订单量
        alpha, beta, gamma (float): 权重系数

    返回:
        float: 预测当日订单量
    """
    return alpha * ad_spend + beta * discount_rate + gamma * prev_orders


@mcp.tool
def simulate_dissolved_oxygen(t, a=1.0, b=0.1, c=0.5, d=2.0):
    """
    水产养殖溶解氧动态模型：
    DO(t) = a * exp(-b*t) + c * sin(d*t)

    参数:
        t (float): 时间（小时）
        a, b, c, d (float): 模型参数

    返回:
        float: t 时刻的溶解氧浓度
    """
    return a * math.exp(-b * t) + c * math.sin(d * t)

if __name__ == "__main__":
    mcp.run(transport="sse", port=8090)