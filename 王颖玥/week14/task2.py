import numpy as np
import math
from fastmcp import FastMCP
import asyncio

mcp = FastMCP("Formula MCP Server")

@mcp.tool
async def get_fish_population(P_t: float, r: float, K: float, S: float) -> float:
    """
    在水产养殖管理中，科学预测和控制鱼类种群数量对于实现可持续发展和经济效益最大化具
    有重要意义。为实现这一目标，构建一个动态模型以描述鱼类种群随时间变化的趋势是关键。
    该模型不仅考虑种群自身的自然增长特性，还引入人为干预因素（如鱼苗的定期投放），
    从而更贴近实际养殖环境。通过模型的建立与分析，可以对未来的种群规模进行预测，并为
    科学决策提供依据。
    :param P_t: 表示第 $ t $ 个时间步的鱼类种群数量（单位：千尾）；
    :param r: 为种群的内禀增长率，反映种群在理想条件下的自然增长能力；
    :param K: 表示环境承载力，即养殖环境中所能支持的最大种群数量；
    :param S: 为每时间步新增投放的鱼苗数量，作为模型的外部输入变量。
    :return: 第 t+1 个时间步的鱼类种群数量
    """
    return P_t + r * P_t * (1 - P_t / K) + S


@mcp.tool
async def get_order(discount_rate: float, prev_orders: float, ad_spend: float):
    """
    在电子商务运营中，准确预测每日订单增长量对于库存管理、资源配置和营销策略优化具有
    重要意义。为了构建一个简洁且具有解释性的预测模型，我们考虑从三个关键业务驱动因素
    出发：广告支出（ad_spend）、当日折扣力度（discount_rate)、前一日订单数量（prev_orders）。
    这些变量分别代表了市场推广投入、价格刺激效应以及订单增长的惯性趋势。基于上述变量，
    我们设计了一个用于预测当日订单数量的一阶线性差分方程模型。该模型以历史数据为基础，
    通过量化各因素对订单增长的影响程度，建立输入变量与输出目标之间的动态关系。
    模型具有良好的可解释性和一定的预测能力，适用于短期订单趋势模拟与敏感性分析。
    :param ad_spend: 广告支出
    :param discount_rate: 当日折扣力度
    :param prev_orders: 前一日订单数量
    :return: 当日订单数量的预测值
    """
    a = 0.05
    b = 100
    c = 0.7
    return a * ad_spend + b * discount_rate + c * prev_orders


@mcp.tool
async def get_crop_yield(F: float, I: float, T: float) -> float:
    """
    在农业科研领域，准确预测作物产量对于制定种植策略、优化资源配置以及提升农业生产效
    率具有重要意义。本模型旨在构建一个基于关键环境与土壤因素的确定性方程，用于估算单
    位面积上的作物产量。该模型综合考虑了土壤肥力、灌溉量以及气温对作物生长的影响，适
    用于在可控环境条件下对产量进行定量分析和趋势预测。
    :param F: 土壤肥力指数
    :param I: 每周灌溉量（mm/week）
    :param T: 平均气温（℃）
    :return: 单位面积作物产量（kg/ha）
    """
    a = 0.5
    b = 0.3
    c = 0.02
    return a * F + b * I - c * T * T


@mcp.tool
async def get_combustion_efficiency(HV: float) -> float:
    """
    在能源化工领域，燃烧过程的效率是评估燃料性能和系统优化的关键指标之一。燃烧效率受
    到多种因素的影响，其中燃料的热值（Heating Value, HV）是一个核心变量。为了建立
    热值与燃烧效率之间的定量关系，本文构建了一个基于经验数据的确定性模型。该模型适用
    于热值范围在40至50MJ/kg之间的燃料，旨在为燃烧系统的设计与运行提供理论支持和数值预测。
    :param HV: 燃料热值
    :return: 燃烧效率
    """
    if HV < 40 or HV > 50:
        raise ValueError(f"参数HV非法：只能介于40-50之间")
    return 0.85 + 0.005 * (HV - 40)


@mcp.tool
async def get_fuel_consumption(weight: float) -> float:
    """
    在汽车制造与设计领域，评估汽车燃油消耗水平是一项关键任务。为了量化汽车重量对百公
    里油耗的影响，构建了一个基于线性关系的预测模型。该模型旨在反映车辆重量与油耗之间
    的基本趋势，为汽车设计和能源效率优化提供参考依据。
    :param weight: 汽车重量（千克）
    :return: 百公里油耗（升/100km）
    """
    return 0.001 * weight + 5


@mcp.tool
async def get_satisfaction(weather: float, crowd_level: float, amenities: float, safety: float, scenery: float) -> float:
    """
    在旅游休闲领域，游客的满意度受到多种因素的综合影响。为了量化评估这些因素对旅游体
    验的影响程度，构建了一个基于多维度输入的代数模型。该模型旨在通过五个关键指标——
    天气舒适度、人流量、设施完善度、安全性和风景优美度，综合计算出一个“旅游满意度评
    分”，从而为旅游目的地的优化与推荐提供数据支持。
    :param weather: 天气
    :param crowd_level: 人流量
    :param amenities: 设施完善度
    :param safety: 安全性
    :param scenery: 风景优美度
    :return satisfaction_rate 旅游满意度
    """
    return 0.2 * weather - 0.15 * crowd_level + 0.2 * amenities + 0.25 * safety + 0.3 * scenery


@mcp.tool
async def get_conversion_rate(P: float, r: float, t: int) -> float:
    """
    在金融服务业中，确定性模型被广泛应用于预测未来资金价值，尤其在固定利率环境下，该
    类模型能够提供精确的本息增长估算。本建模示例聚焦于三年期定期存款，在年利率固定为
    5%的前提下，通过复利机制计算客户本金在存款期满后的本息总额。该模型可用于银行产
    品定价、客户收益预测以及财务规划等场景，具有较高的实用性和可扩展性。
    :param P: 表示初始存入的本金
    :param r: 为年利率，设定为5%
    :param t: 为存款年限，设定为3年
    :return: 三年后账户的本息总额(A)
    """
    r = 0.05
    t = 3
    return P * (1 + r) ** t


@mcp.tool
async def get_milk(quality: float, T: float, R:  float, t: float):
    """
    本模型旨在模拟和预测奶牛在特定饲养条件下的日均产奶量（单位：kg）。通过综合考虑饲料质量、
    饮水时间、健康状况以及环境温度等关键影响因素，建立一个具有统计合理性和现实适应性的估算模型。
    模型中引入了随机扰动项，以反映实际生产中可能存在的个体差异、测量误差以及其他未被显性变量捕
    捉的生物学波动。该模型可用于牧场管理中的饲养策略评估与优化，为决策提供数据支持。
    :param quality: 饲料质量
    :param T: 气温
    :param R：健康评分
    :param t：饮水时间
    :return: 日均产奶量（单位：kg）
    """
    return 20 + 1.5 * quality - 0.3 * (T - 20) + 0.05 * R + 0.4 * t


@mcp.tool
async def get_mastery_score(time: float) -> float:
    """
    在教育培训领域，评估学生的学习效果是教学管理中的关键环节。为了量化学习时间与知识
    掌握程度之间的关系，建立一个简洁有效的预测模型具有重要意义。本模型旨在通过学生每
    日学习时间（单位：小时）来预测其知识掌握度得分（范围：0~100）。该模型可用于
    教学效果的初步评估、学习计划的优化以及个性化教学策略的制定。
    :param time: 学习时间（单位：小时）
    :return: 掌握度得分score
    """
    return 5 + 8 * time


@mcp.tool
async def get_average_travel_time(v: float, w: float, p: int, n: int):
    """
    在交通运输领域，准确预测交通流的通行时间对于交通管理和出行规划具有重要意义。该模
    型旨在基于道路的基本属性（如限速、车道数量）以及动态交通状态（如车辆密度）和外部
    环境因素（如天气条件），对某一段道路的平均通行时间进行量化预测。
    模型的核心思想是结合交通流的基本理论，采用简化的积分建模方法，模拟在不同交通负荷
    和环境干扰下，车辆通过特定路段所需的时间变化。通过引入密度影响因子、车道数量调节
    因子以及天气影响因子，模型能够在一定程度上反映实际交通运行状态的复杂性和非线性特
    征。
    :param v：道路限速（公里/小时）
    :param w：天气影响因子（无量纲，基准值为1）
    :param p：车辆密度（辆/公里）
    :param n：车道数量（条）
    :return: 平均通行时间T（分钟）
    """
    return (1 / (v / w)) * (1 + np.log(1 + p / 100)) * (1 / np.sqrt(n)) * 60


if __name__ == "__main__":
    try:
        mcp.run(transport="sse", host="localhost", port=8900)
    except Exception as e:
        print(f"服务器启动失败{str(e)}")
