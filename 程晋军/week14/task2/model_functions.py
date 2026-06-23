#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Markdown 文档中提取的模型函数

该模块包含了从多个 Markdown 文档中提取的数学模型函数，
每个函数都对应一个具体的应用场景。
"""

import math
import numpy as np


def dissolved_oxygen(t, a, b, c, d):
    """
    计算溶解氧浓度随时间的变化
    
    在水产养殖系统中，溶解氧（DO）是影响水生生物健康和生长的关键环境因子之一。
    该模型融合了指数衰减项和周期性扰动项，分别反映溶解氧的自然消耗过程和环境因素引起的波动特性。
    
    公式: DO(t) = a * exp(-b * t) + c * sin(d * t)
    
    Args:
        t: 时间
        a: 初始溶解氧释放量，反映系统初始状态下的氧含量
        b: 溶解氧的衰减系数，刻画其随时间自然下降的速率
        c: 环境扰动的振幅，体现外部周期性因素对DO浓度的影响强度
        d: 环境扰动的频率，反映扰动周期的快慢
    
    Returns:
        float: t时刻的溶解氧浓度
    """
    return a * math.exp(-b * t) + c * math.sin(d * t)


def orders_prediction(ad_spend, discount_rate, prev_orders, alpha=0.05, beta=100, gamma=0.7):
    """
    预测电商每日订单增长量
    
    用于预测当日订单数量的一阶线性差分方程模型。
    综合考虑广告支出、折扣力度和前一天订单数对当前订单量的影响。
    
    公式: orders_t = α * ad_spend + β * discount_rate + γ * prev_orders
    
    Args:
        ad_spend: 广告支出
        discount_rate: 当日折扣力度
        prev_orders: 前一天订单数量
        alpha: 广告支出对订单量的敏感系数 (默认 0.05)
        beta: 折扣率对订单量的放大系数 (默认 100)
        gamma: 前一日订单数量对当前日订单趋势的惯性影响 (默认 0.7)
    
    Returns:
        float: 预测的当日订单数量
    """
    return alpha * ad_spend + beta * discount_rate + gamma * prev_orders


def moisture_content(t, M0, k):
    """
    计算食品干燥过程中水分含量随时间的变化
    
    水分含量随时间的变化关系
    
    公式: M(t) = M0 * exp(-k * t)
    
    Args:
        t: 干燥时间
        M0: 初始水分含量
        k: 水分蒸发速率常数
    
    Returns:
        float: t时刻食品的水分含量
    """
    return M0 * math.exp(-k * t)


def evaporated_amount(T, M0, k):
    """
    计算食品干燥过程中累计水分蒸发量
    
    公式: Evaporated(T) = M0 * (T + (exp(-k * T) - 1) / k)
    
    Args:
        T: 干燥时间
        M0: 初始水分含量
        k: 水分蒸发速率常数
    
    Returns:
        float: 在时间T内食品的总水分损失量
    """
    return M0 * (T + (math.exp(-k * T) - 1) / k)


def crop_yield(F, I, T, a, b, c):
    """
    估算单位面积上的作物产量
    
    公式: Y = a * F + b * I - c * T^2
    
    Args:
        F: 土壤肥力指数
        I: 每周灌溉量 (mm/week)
        T: 平均气温 (°C)
        a: 土壤肥力对产量的贡献系数
        b: 灌溉量对产量的贡献系数
        c: 气温对产量的抑制系数
    
    Returns:
        float: 单位面积作物产量 (kg/ha)
    """
    return a * F + b * I - c * (T ** 2)


def student_score(x1, x2, x3, x4, w1, w2, w3, w4, alpha=1.0, beta=0.0):
    """
    评估学生学习效果
    
    公式: Score = 100 / (1 + exp(-α(w1*x1 + w2*x2 + w3*x3 + w4*x4 - β)))
    
    Args:
        x1: 学习时长 (小时)
        x2: 出勤率 (百分比)
        x3: 平时测验平均分 (百分比)
        x4: 课堂参与度 (1~5分)
        w1: 学习时长的权重系数
        w2: 出勤率的权重系数
        w3: 平时测验平均分的权重系数
        w4: 课堂参与度的权重系数
        alpha: 控制S型曲线的陡峭程度
        beta: 控制曲线在横轴上的平移位置
    
    Returns:
        float: 学生学习效果评分 (0-100)
    """
    linear_combination = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4
    exponent = -alpha * (linear_combination - beta)
    return 100 / (1 + math.exp(exponent))


def difference_equation_model(x1_t, y_t_minus_1, y_t_minus_2, x2_t, x3_t, a, b, c, d):
    """
    差分方程模型
    
    公式: y_t = a*x1_t + b*y_{t-1} + c*y_{t-2} + d*x2_t*x3_t
    
    Args:
        x1_t: 当前时刻的第一个输入变量
        y_t_minus_1: 前一时刻的输出值
        y_t_minus_2: 前两时刻的输出值
        x2_t: 当前时刻的第二个输入变量
        x3_t: 当前时刻的第三个输入变量
        a: x1_t的权重系数
        b: y_t_minus_1的权重系数
        c: y_t_minus_2的权重系数
        d: x2_t*x3_t的权重系数
    
    Returns:
        float: 当前时刻的输出值
    """
    return a * x1_t + b * y_t_minus_1 + c * y_t_minus_2 + d * x2_t * x3_t


def deterministic_model(x):
    """
    确定性模型 (二次函数)
    
    公式: y = 2*x^2 + 3*x + 1
    
    Args:
        x: 输入变量
    
    Returns:
        float: 输出结果
    """
    return 2 * (x ** 2) + 3 * x + 1


def influence_model(content_quality, channels, engagement, time):
    """
    传播影响力模型
    
    公式: Influence = content_quality * channels * engagement * time
    
    Args:
        content_quality: 内容质量
        channels: 传播渠道数量
        engagement: 受众参与度
        time: 传播持续时间
    
    Returns:
        float: 综合影响力得分
    """
    return content_quality * channels * engagement * time


def cattle_population(N_t, r, K):
    """
    牛群数量演化模型 (逻辑斯蒂增长模型)
    
    公式: N_{t+1} = N_t + r*N_t*(1 - N_t/K)
    
    Args:
        N_t: 第t年的牛群数量
        r: 年增长率
        K: 环境承载能力
    
    Returns:
        float: 第(t+1)年的牛群数量
    """
    return N_t + r * N_t * (1 - N_t / K)


def linear_difference_equation(y_t_minus_1, x1_t, x2_t, x3_t, x4_t, x5_t, a, b, c, d, e):
    """
    一阶线性差分方程
    
    公式: y_t = a*y_{t-1} + b*x1_t - c*x2_t + d*x3_t + e*(x4_t - x5_t)
    
    Args:
        y_t_minus_1: 前一时刻的系统状态
        x1_t: 当前时刻的第一个输入变量
        x2_t: 当前时刻的第二个输入变量
        x3_t: 当前时刻的第三个输入变量
        x4_t: 当前时刻的第四个输入变量
        x5_t: 当前时刻的第五个输入变量
        a: y_t_minus_1的权重系数
        b: x1_t的权重系数
        c: x2_t的权重系数
        d: x3_t的权重系数
        e: (x4_t - x5_t)的权重系数
    
    Returns:
        float: 当前时刻的系统状态输出
    """
    return a * y_t_minus_1 + b * x1_t - c * x2_t + d * x3_t + e * (x4_t - x5_t)


# 测试示例
if __name__ == "__main__":
    # 测试溶解氧模型
    print("溶解氧模型测试:")
    do_value = dissolved_oxygen(t=1.0, a=5.0, b=0.1, c=2.0, d=0.5)
    print(f"DO(1.0) = {do_value:.2f}")
    
    # 测试订单预测模型
    print("\\n订单预测模型测试:")
    orders = orders_prediction(ad_spend=1000, discount_rate=0.1, prev_orders=50)
    print(f"预测订单数: {orders:.0f}")
    
    # 测试学生评分模型
    print("\\n学生评分模型测试:")
    score = student_score(x1=10, x2=90, x3=85, x4=4, w1=0.3, w2=0.2, w3=0.4, w4=0.1)
    print(f"学生评分: {score:.2f}")