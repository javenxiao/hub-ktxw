import asyncio
from typing import Annotated, Union

import numpy as np
from fastmcp import FastMCP, Client

'''
解析10个pdf文档，并定义10个对应的mcp tool，可以通过numpy 计算，也可以通过sympy计算

:prompt
    你是一名资深python开发者，对每个文件用python的numpy写完整的单个函数实现，确保能运行；函数名用get_开头，需要见名知意；需要添加函数作用、入参、出参的标准注释，不需要运行示例代码；函数上一行添加@mcp.tool，入参的注释使用Annotated标明，入参的注释中变量与函数入参保持一致；函数的注释只需注明函数功能
样例
@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://www.baidu.com?text={address_text}", timeout=5).json()["result"]
    except:
        return []
'''

mcp = FastMCP("Python SSE Server", "Simple in")


@mcp.tool
def get_dissolved_oxygen_concentration(
        time: Annotated[float, "时间变量，可以是单个时间点或时间序列"],
        initial_oxygen: Annotated[float, "初始溶解氧释放量，反映系统初始状态下的氧含量"],
        decay_rate: Annotated[float, "溶解氧衰减系数，刻画其随时间自然下降的速率"],
        amplitude: Annotated[float, "环境扰动振幅参数，体现外部周期性因素对DO浓度的影响强度"],
        frequency: Annotated[float, "环境扰动频率参数，反映扰动周期的快慢"]
) -> float:
    """计算水产养殖系统中溶解氧浓度随时间变化的数值解"""
    return initial_oxygen * np.exp(-decay_rate * time) + amplitude * np.sin(frequency * time)


@mcp.tool
def get_daily_orders_prediction(
        ad_spend_t: Annotated[float, "当日广告支出金额"],
        discount_rate_t: Annotated[float, "当日折扣率(0-1之间的小数)"],
        prev_orders_t: Annotated[float, "前一日订单数量"]
) -> float:
    """基于广告支出、折扣率和历史订单量预测当日订单数量的线性模型"""
    # 模型参数
    alpha = 0.05  # 广告支出敏感系数
    beta = 100  # 折扣率放大系数
    gamma = 0.7  # 订单惯性系数

    # 计算当日订单预测值
    return alpha * ad_spend_t + beta * discount_rate_t + gamma * prev_orders_t


@mcp.tool
def get_crop_yield(soil_fertility: Annotated[float, "土壤肥力指数"],
                   irrigation_amount: Annotated[float, "每周灌溉量(mm/week)"],
                   temperature: Annotated[float, "平均气温(℃)"]) -> float:
    """基于土壤肥力、灌溉量和气温预测单位面积作物产量"""
    a = 2.5  # 土壤肥力系数
    b = 1.8  # 灌溉量系数
    c = 0.3  # 温度抑制系数

    return a * soil_fertility + b * irrigation_amount - c * np.power(temperature, 2)


@mcp.tool
def get_cumulative_evaporation(initial_moisture: Annotated[float, "初始水分含量M₀"],
                               evaporation_rate: Annotated[float, "水分蒸发速率常数k"],
                               drying_time: Annotated[float, "总干燥时间T"]) -> Union[
    float, np.ndarray]:
    """计算在给定干燥时间内的累计水分蒸发量"""
    # 处理k=0的特殊情况（避免除零错误）
    evaporation_rate_safe = np.where(evaporation_rate == 0, 1e-10, evaporation_rate)

    # 计算累计水分蒸发量：M₀ * (T + (e^(-kT) - 1) / k)
    return initial_moisture * (drying_time + (np.exp(-evaporation_rate_safe * drying_time) - 1) / evaporation_rate_safe)


@mcp.tool
def get_student_performance_score(
        study_hours: Annotated[float, "学习时长(小时)"],
        attendance_rate: Annotated[float, "出勤率(百分比)"],
        quiz_score: Annotated[float, "平时测验平均分(百分比)"],
        participation_level: Annotated[float, "课堂参与度(1~5分)"]
) -> float:
    """基于学习时长、出勤率、测验成绩和课堂参与度预测学生综合表现分数"""
    # 权重系数设置
    w1, w2, w3, w4 = 0.3, 0.25, 0.3, 0.15  # 学习时长、出勤率、测验成绩、参与度的权重

    # Sigmoid函数参数
    alpha = 0.1  # 控制曲线陡峭程度
    beta = 5.0  # 控制曲线平移位置

    # 课堂参与度线性映射 (1~5分映射到0~100分)
    mapped_participation = (participation_level - 1) * 25  # 映射到0~100范围

    # 计算加权线性组合
    linear_combination = (w1 * study_hours +
                          w2 * attendance_rate +
                          w3 * quiz_score +
                          w4 * mapped_participation - beta)

    # 应用Sigmoid函数并缩放到0-100分
    performance_score = 100 / (1 + np.exp(-alpha * linear_combination))

    return performance_score


@mcp.tool
def get_time_series_prediction(
        a: Annotated[float, "模型参数a，调节x1对输出的影响权重"],
        b: Annotated[float, "模型参数b，调节y_{t-1}对输出的影响权重"],
        c: Annotated[float, "模型参数c，调节y_{t-2}对输出的影响权重"],
        d: Annotated[float, "模型参数d，调节x2和x3交互项对输出的影响权重"],
        y_t_minus_1: Annotated[float, "前一个时间步的输出值y_{t-1}"],
        y_t_minus_2: Annotated[float, "前两个时间步的输出值y_{t-2}"],
        x1_t: Annotated[float, "当前时刻的第一个输入变量x1_t"],
        x2_t: Annotated[float, "当前时刻的第二个输入变量x2_t"],
        x3_t: Annotated[float, "当前时刻的第三个输入变量x3_t"]
) -> float:
    """基于差分方程的时间序列预测模型，考虑外部输入与内部状态反馈"""
    return a * x1_t + b * y_t_minus_1 + c * y_t_minus_2 + d * x2_t * x3_t


@mcp.tool
def get_quadratic_deterministic_output(
        x: Annotated[float, "输入变量x"]
) -> float:
    """基于二次函数关系的确定性模型计算输出"""
    return 2 * np.power(x, 2) + 3 * x + 1


@mcp.tool
def get_cultural_influence_score(
        content_quality: Annotated[float, "内容质量指标"],
        channels: Annotated[float, "传播渠道数量"],
        engagement: Annotated[float, "受众参与度"],
        time: Annotated[float, "传播持续时间"]
) -> float:
    """计算文化传播项目的综合影响力得分"""
    return content_quality * channels * engagement * time


@mcp.tool
def get_cattle_population_growth(
        N_t: Annotated[float, "当前年份牛群数量N_t"],
        r: Annotated[float, "年增长率r"],
        K: Annotated[float, "环境承载能力K"]
) -> float:
    """基于逻辑斯蒂增长模型预测下一年牛群数量"""
    return N_t + r * N_t * (1 - N_t / K)


@mcp.tool
def get_multi_input_dynamic_system(
        a: Annotated[float, "模型参数a，调节y_{t-1}对当前状态的影响权重"],
        b: Annotated[float, "模型参数b，调节输入变量x1_t的影响权重"],
        c: Annotated[float, "模型参数c，调节输入变量x2_t的影响权重"],
        d: Annotated[float, "模型参数d，调节输入变量x3_t的影响权重"],
        e: Annotated[float, "模型参数e，调节输入变量x4_t和x5_t差值的影响权重"],
        y_t_minus_1: Annotated[float, "前一时刻的系统状态y_{t-1}"],
        x1_t: Annotated[float, "当前时刻的第一个外部输入变量x1_t"],
        x2_t: Annotated[float, "当前时刻的第二个外部输入变量x2_t"],
        x3_t: Annotated[float, "当前时刻的第三个外部输入变量x3_t"],
        x4_t: Annotated[float, "当前时刻的第四个外部输入变量x4_t"],
        x5_t: Annotated[float, "当前时刻的第五个外部输入变量x5_t"]
) -> float:
    """多变量驱动的一阶线性差分方程系统状态预测"""
    return a * y_t_minus_1 + b * x1_t - c * x2_t + d * x3_t + e * (x4_t - x5_t)


async def ping():
    async with Client(mcp) as client:
        tools_list = await client.list_tools()
        tls={}
        for i,tool in enumerate(tools_list):
            tls[i]=tool.name
        print(tls)

if __name__ == '__main__':
    asyncio.run(ping())
    mcp.run(transport="sse", port=8900)
