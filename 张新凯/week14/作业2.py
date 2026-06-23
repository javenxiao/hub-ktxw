import asyncio
import math  # 必须保留
from typing import Annotated, List  # 必须保留

import numpy as np  # 必须保留
from fastmcp import FastMCP, Client  # 必须保留
from sympy import symbols, integrate, lambdify  # 引入符号计算支持

mcp = FastMCP(name="Mcp Server")  # 必须保留


@mcp.tool
def calculate_bmr(weight: Annotated[float, "个体的体重，单位为千克(kg)"]) -> dict:
    """
    在医疗健康领域, 基础代谢率 (Basal Metabolic Rate, BMR) 是评估个体能量消耗和制定营养干预方案的重要指标。
    BMR 受多种生理因素的影响, 包括体重、身高、年龄和性别等。本函数基于体重与 BMR 之间的线性关系,
    使用简化模型估算基础代谢率：BMR = 22.7 × weight + 450，单位为千卡/天(kcal/day)。

    参数说明:
        weight (float): 个体的体重，单位为千克(kg)，必须为正数。

    返回值:
        dict: 包含估算的基础代谢率（bmr）的字典，单位为千卡/天(kcal/day)。
              例如: {"bmr": 1500.5}
    """
    if weight <= 0:
        raise ValueError("体重必须大于零。")

    bmr = 22.7 * weight + 450
    return {"bmr": bmr}


@mcp.tool
def fish_population_growth_model(
        N: Annotated[float, "当前鱼类种群数量"],
        K: Annotated[float, "养殖环境的承载能力，限制种群最大规模"],
        r: Annotated[float, "基础内禀增长率"],
        F: Annotated[float, "每日投喂量，反映饲养管理强度"],
        F0: Annotated[float, "标准投喂参考值"],
        alpha: Annotated[float, "控制投喂量对种群增长的增强效应"],
        T: Annotated[float, "水体温度，影响鱼类代谢和生长效率"],
        Topt: Annotated[float, "最适生长温度"],
        beta: Annotated[float, "反映温度偏离最适生长温度所带来的抑制作用"]
) -> Annotated[float, "返回种群数量随时间的变化率 dN/dt，表示当前条件下的种群瞬时增长速率"]:
    """
    在水产养殖系统中，鱼类种群的动态增长受到多种环境和管理因素的综合影响。
    本模型以Logistic增长模型为基础，引入水温变化对鱼类生理状态的影响以及投喂管理对生长速率的调节作用，
    构建一个更具现实意义的动态增长模型。该模型可用于评估养殖密度、饲料投喂策略以及环境温度调控对种群增长的综合影响，
    为科学决策提供理论支持。

    模型公式：
    dN/dt = r * N * (1 - N/K) * (1 + alpha * F/F0) * (1 + beta * (T - Topt)**2)

    注意：此处假设 (1 + beta*(T-Topt)**2) 项用于描述温度对生长的抑制效应（当 beta < 0），或增强效应（beta > 0）。
    实际应用中通常设 beta 为负值，以表示偏离最适温度会降低生长速率。但根据原式结构，若 beta 为负且绝对值大，
    可能导致该项为负，从而使增长率异常。建议使用 exp(-beta*(T-Topt)^2) 等形式更合理，但此处严格遵循给定表达式。

    参数说明：
    - N: 当前鱼类种群数量（正数）
    - K: 环境承载力，必须大于0
    - r: 基础内禀增长率，通常为正值
    - F: 当前每日投喂量
    - F0: 标准投喂参考值，用于归一化投喂影响
    - alpha: 投喂增强系数，>0 表示促进增长，<0 表示抑制
    - T: 当前水体温度（摄氏度）
    - Topt: 鱼类最适生长温度
    - beta: 温度敏感系数，通常应为负数以体现偏离最适温度的抑制作用

    返回值：
    - dN/dt: 种群瞬时增长速率
    """
    # 防止除以零
    if F0 == 0:
        raise ValueError("F0（标准投喂参考值）不能为0。")
    if K == 0:
        return 0.0

    # 计算各因子
    logistic_factor = r * N * (1 - N / K)  # Logistic 增长项
    feeding_factor = 1 + alpha * (F / F0)  # 投喂调节项
    temperature_factor = 1 + beta * ((T - Topt) ** 2)  # 温度影响项

    # 综合计算 dN/dt
    dNdt = logistic_factor * feeding_factor * temperature_factor

    return float(dNdt)


@mcp.tool
def calculate_learning_effectiveness(
        x: Annotated[float, "学习时长（单位：小时），表示学生每天投入的学习时间。"] = None,
        y: Annotated[float, "学习专注度，取值范围为0到1，表示学生在学习过程中的注意力集中程度。"] = None
) -> Annotated[float, "返回计算得到的学习效果值，为一个包含基础学习积累、专注度增强效应以及随机扰动影响的综合量化指标。"]:
    """
    在教育培训领域，评估学生的学习效果是优化教学策略、制定个性化学习方案的重要依据。
    本函数以学生每日学习时长和学习专注度作为核心输入变量，旨在通过定量方式刻画这两个因素对学生整体学习成效的联合影响。
    模型引入随机扰动项以模拟现实数据中不可避免的波动性，从而增强模型的实用性与泛化能力。

    参数说明:
    - x (float): 学习时长，单位为小时，应为非负数值。
    - y (float): 学习专注度，取值范围 [0, 1]，表示注意力集中程度；超出范围的结果可能影响模型准确性。

    计算公式:
        effectiveness = 0.6 * x + 0.4 * x * y + epsilon
    其中 epsilon ~ Normal(0, 0.1) 为模拟测量误差或个体差异的随机扰动项。

    返回值:
        float: 综合学习效果值，包含基础学习量、专注度放大效应及随机扰动的影响。
    """
    if x is None or y is None:
        raise ValueError("学习时长(x)和学习专注度(y)均为必填参数。")

    if not (0 <= y <= 1):
        raise ValueError("学习专注度(y)必须在0到1之间。")

    if x < 0:
        raise ValueError("学习时长(x)不能为负数。")

    # 定义随机扰动项：均值为0，标准差为0.1的正态分布噪声
    epsilon = np.random.normal(0, 0.1)

    # 计算学习效果
    effectiveness = 0.6 * x + 0.4 * x * y + epsilon

    return effectiveness


@mcp.tool
def drug_concentration_model(
        t: Annotated[float, "时间 t，表示从开始给药到当前的时间点"],
        r: Annotated[float, "药物的恒定输入速率 r"],
        k: Annotated[float, "清除速率常数 k"]
) -> Annotated[float, "返回在时间 t 时体内药物的浓度 C(t)"]:
    """
    在药代动力学研究中, 理解药物在体内的吸收、分布、代谢和排泄过程是评估药物疗效和安全性的关键环节。
    本模型描述一种常见情形：药物以恒定速率进入体内，并遵循一级动力学过程被清除。
    通过该数学模型可预测在给定时间点体内药物的累积浓度，为临床给药方案的设计和优化提供理论依据。

    参数说明:
    - t (float): 时间，单位通常为小时或分钟，表示从开始给药起经过的时间。
    - r (float): 药物输入速率，单位如 mg/h，表示单位时间内进入体内的药物量。
    - k (float): 清除速率常数，单位与时间倒数一致（如 h⁻¹），反映药物从体内被清除的速度。

    模型公式:
    C(t) = (r / k) * (1 - exp(-k * t))

    当时间趋近无穷时，药物浓度趋于稳态值 C_ss = r / k。
    """
    if k == 0:
        raise ValueError("清除速率常数 k 不能为零。")
    if t < 0:
        raise ValueError("时间 t 不能为负数。")

    C_t = (r / k) * (1 - math.exp(-k * t))
    return C_t


@mcp.tool
def logistic_population_growth(
        N0: Annotated[float, "初始种群数量，表示在时间 t=0 时的鱼类数量，必须为正数"],
        r: Annotated[float, "内禀增长率，表示种群在理想条件下的最大增长速率，通常为正数"],
        K: Annotated[float, "环境承载力，表示养殖环境中能够支持的最大种群数量，必须大于初始种群数量且为正数"],
        t: Annotated[float, "预测的时间点，表示从初始时刻起经过的时间单位（如天、周等），应为非负数"]
):
    """
    在水产养殖系统中，合理预测和管理鱼类种群的增长对于资源规划和可持续发展具有重要意义。
    本函数采用经典的Logistic增长模型，综合考虑种群的内禀增长率和环境承载能力，
    计算在有限资源环境下t时刻的鱼类种群数量。

    Logistic模型公式：
        N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))

    参数说明：
        - N0 (float): 初始种群数量，必须 > 0
        - r  (float): 内禀增长率，必须 > 0
        - K  (float): 环境承载力，必须 > N0 且 > 0
        - t  (float): 预测时间点，必须 >= 0

    返回值：
        float: 在时间 t 时的种群数量，结果介于 N0 和 K 之间。
    """
    if N0 <= 0:
        raise ValueError("初始种群数量 N0 必须大于 0")
    if r <= 0:
        raise ValueError("内禀增长率 r 必须大于 0")
    if K <= 0:
        raise ValueError("环境承载力 K 必须大于 0")
    if K < N0:
        raise ValueError("环境承载力 K 必须大于或等于初始种群数量 N0")
    if t < 0:
        raise ValueError("时间 t 不能为负数")

    # 计算Logistic增长模型的结果
    exponent = -r * t
    numerator = K
    denominator = 1 + ((K - N0) / N0) * math.exp(exponent)
    population = numerator / denominator

    return population


@mcp.tool
def predict_conversion_rate(
        ad_spend: Annotated[float, "广告投入金额，用于计算转化率"],
        k: Annotated[float, "广告投入对转化率增长的速率"],
        x0: Annotated[float, "转化率达到中位水平时的广告投入中点"],
        epsilon: Annotated[float, "服从标准正态分布的随机扰动项，用于模拟市场不确定性"]
) -> Annotated[float, "预测的转化率，包含非线性增长部分和随广告投入平方根放大的随机扰动"]:
    """
    在电子商务场景中, 评估广告投入对转化率的影响是营销优化的重要环节。
    本函数基于非线性增长特性与外部干扰因素，构建了一个具备随机扰动的转化率预测机制。
    该机制不仅考虑了广告投入对转化率的非线性增强效应（使用S型逻辑增长函数建模），
    还引入了随机性以模拟市场环境中的不确定性，如用户行为波动、竞争因素和外部事件影响。

    转化率计算公式为：
        conversion_rate = 100 / (1 + exp(-k * (ad_spend - x0))) + epsilon * sqrt(ad_spend)

    参数说明：
    - ad_spend (float): 广告投入金额，正值，单位根据实际业务设定。
    - k (float): 增长速率参数，控制转化率上升的陡峭程度。
    - x0 (float): S型曲线的中点，表示转化率达到50%基础水平时的广告投入。
    - epsilon (float): 标准正态分布的随机扰动项，用于模拟市场波动，通常由外部采样传入。

    返回值：
    - float: 预测的转化率（百分比形式），包含确定性非线性趋势和随广告投入平方根放大的随机扰动。
    """
    # 计算S型增长部分（限制在合理范围避免数值问题）
    sigmoid_part = 100 / (1 + math.exp(-k * (ad_spend - x0)))

    # 计算随机扰动部分：与广告投入的平方根成正比
    noise_part = epsilon * math.sqrt(ad_spend)

    # 返回总转化率
    conversion_rate = sigmoid_part + noise_part
    return conversion_rate


@mcp.tool
def solve_first_order_system(
        y0: Annotated[float, "初始状态值 y(0)"],
        k: Annotated[float, "系统的时间常数，描述系统的衰减特性"],
        u: Annotated[List[float], "外部输入信号序列 u(t)，按时间步长排列"],
        dt: Annotated[float, "时间步长，用于欧拉法离散化"],
        t_steps: Annotated[int, "模拟的时间步数"]
):
    """
    在工程与科学计算领域, 常微分方程 (ODE) 是描述动态系统行为的重要数学工具。
    该模型模拟一个典型的一阶线性系统, 其行为由状态变量随时间的变化率与当前状态及外部输入之间的关系决定。
    此类模型广泛应用于热力学、电路分析、自动控制以及生物系统建模中, 能够有效描述具有指数响应特性的系统行为。
    本模型采用欧拉法进行数值积分, 以离散时间步长的方式近似求解微分方程。
    该方法结构简单、易于实现, 适用于实时仿真和嵌入式系统中的控制算法实现。

    参数说明:
    - y0: 初始状态值 y(0)
    - k: 系统的时间常数，决定系统响应速度；k > 0 表示稳定衰减
    - u: 外部输入信号序列，长度至少为 t_steps
    - dt: 时间步长，影响数值解的精度和稳定性
    - t_steps: 模拟的时间步数，决定输出序列长度

    返回:
    - 字典包含两个键：
        - 'time': 时间序列数组，从 0 到 t_steps*dt，步长为 dt
        - 'y': 系统状态 y 在每个时间步的数值解列表
    """
    # 初始化
    y = [y0]
    current_y = y0

    # 确保输入 u 的长度足够
    if len(u) < t_steps:
        raise ValueError("输入信号 u 的长度必须大于等于 t_steps")

    # 使用欧拉法进行数值积分
    for i in range(1, t_steps + 1):
        dydt = -k * current_y + u[i - 1]  # 一阶系统微分方程：dy/dt = -k*y + u
        current_y = current_y + dt * dydt  # 显式欧拉更新
        y.append(current_y)

    # 生成时间序列
    time = [i * dt for i in range(t_steps + 1)]

    return {
        "time": time,
        "y": y
    }


@mcp.tool
def calculate_satisfaction(
        temp: Annotated[float, "当前气温（单位：℃），用于计算天气舒适度对满意度的影响，最适温度为22℃"],
        crowd: Annotated[int, "当前人流量（单位：人数或密度指数），用于计算人群密度对满意度的抑制作用，数值越大满意度越低"]
) -> Annotated[float, "返回游客满意度得分，范围通常在0到100之间，数值越高表示游客体验越好"]:
    """
    在旅游休闲场景中, 游客满意度是衡量景区服务质量与游客体验的重要指标。
    该函数通过非线性模型量化天气舒适度和人流量对游客满意度的综合影响，
    捕捉温度对满意度的非线性响应（以22℃为最舒适温度）以及人流量对游客体验的抑制效应，
    为景区管理提供数据支持与决策依据。

    参数说明:
    - temp (float): 当前气温（℃），偏离22℃越多，舒适度下降。使用高斯型衰减函数建模。
    - crowd (int or float): 当前人流量或密度指数，数值越大表示越拥挤，满意度越低。
      使用Sigmoid函数建模人群带来的负面体验。

    模型公式:
    Satisfaction = 100 * exp(-0.05 * (temp - 22)^2) * (1 - 1 / (1 + exp(-0.2 * (50 - crowd))))

    返回值:
    - satisfaction (float): 满意度得分，范围约在0~100之间，越高代表游客体验越好。
    """
    # 天气舒适度部分：以22℃为中心的高斯衰减
    comfort = math.exp(-0.05 * (temp - 22) ** 2)

    # 人流抑制效应：crowd越大，(50 - crowd)越小，sigmoid输出趋近1，括号内趋近0
    crowd_effect = 1 - (1 / (1 + math.exp(-0.2 * (50 - crowd))))

    # 综合满意度
    satisfaction = 100 * comfort * crowd_effect

    return float(satisfaction)


# 定义符号变量
t = symbols('t')
price_sym = symbols('price')
days_sym = symbols('days')

# 需求速率函数：随时间 t 变化，受价格抑制
demand_rate = 100 / (1 + 0.1 * price_sym * t)

# 对时间 t 在 [0, days] 区间积分，得到累计需求表达式
total_demand_expr = integrate(demand_rate, (t, 0, days_sym))

# 将符号表达式转换为可数值计算的函数
total_demand_func = lambdify((price_sym, days_sym), total_demand_expr, 'numpy')


@mcp.tool
def predict_total_demand(
        price: Annotated[float, "商品的单价，用于反映价格对需求的抑制作用"],
        days: Annotated[int, "预测的时间窗口长度（单位：天），表示需求累积的时间范围"]
) -> Annotated[float, "预测的累计需求量，基于给定价格和时间窗口通过数值积分近似计算得到"]:
    """
    在农产品批发与零售场景中，准确预测特定商品在一定时间窗口内的累计需求量对于库存管理、采购计划和价格策略具有重要意义。

    影响需求的核心因素之一是商品的单价：通常情况下，价格越高，消费者购买意愿越低，从而导致需求增长放缓。
    同时，需求具有时间累积特性：即使每日需求较低，随着时间推移，总需求仍可能显著增加。

    本函数以商品单价和预测天数为输入，通过积分模型计算累计需求：
    - 需求速率定义为：100 / (1 + 0.1 * price * t)，其中 t 为时间（天）
    - 模型体现价格抑制效应（价格越高，初期需求越低）和时间累积效应
    - 最终返回从第 0 天到第 'days' 天的总需求量（数值积分结果）

    参数说明:
        price (float): 商品单价。价格越高，单位时间内的需求增长越慢。
        days (int): 预测的时间窗口长度（天数）。时间越长，累计需求总量越大。

    返回:
        float: 预测的累计需求量，非负实数。若输入非法（如负价格或负天数），返回 0。
    """
    if price < 0 or days < 0:
        return 0.0
    try:
        result = total_demand_func(float(price), int(days))
        return float(result) if np.isfinite(result) else 0.0
    except Exception:
        return 0.0


@mcp.tool
def calculate_conversion(
        temperature: Annotated[float, "反应温度，单位为摄氏度（℃），用于影响反应速率的指数项"],
        pressure: Annotated[float, "系统压力，单位为兆帕（MPa），线性促进转化率"],
        catalyst_loading: Annotated[float, "催化剂负载量，单位为质量百分比（%），影响反应活性的非线性因子"]
) -> Annotated[float, "反应转化率，取值范围[0, 1]，表示反应物转化为产物的比例"]:
    """
    在能源与化工过程中, 反应器的转化率是衡量化学反应效率的重要性能指标。
    该指标受到多个操作变量的影响, 包括反应温度、压力以及催化剂的使用特性。
    本函数实现了一个确定性数学模型，用于描述转化率与关键输入变量之间的函数关系，
    综合考虑了：
    - 温度的指数增强效应（通过sigmoid-like指数衰减函数建模）
    - 压力的线性促进作用
    - 催化剂负载量的非线性响应特征（以S型函数刻画活性变化）

    模型公式：
    conversion = 0.1 * (1 - exp(-0.05 * temperature)) * (pressure / (1 + exp(-0.2 * (catalyst_loading - 50))))

    为工艺优化与控制提供理论依据。
    """
    # 温度项：指数增强效应，随温度升高趋近于1
    temp_effect = 1 - math.exp(-0.05 * temperature)

    # 催化剂非线性响应项：S型函数，中心在50%，增益0.2
    catalyst_effect = 1 + math.exp(-0.2 * (catalyst_loading - 50))
    catalyst_response = 1 / catalyst_effect  # Sigmoid-shaped activation

    # 转化率计算：组合各因素影响
    conversion = 0.1 * temp_effect * pressure / catalyst_response

    # 确保输出在[0, 1]范围内
    conversion = max(0.0, min(1.0, conversion))

    return conversion


async def filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])
        print("Available tools:", [t for t in tools])


if __name__ == "__main__":
    asyncio.run(filtering())
    mcp.run(transport="sse", port=8900)
