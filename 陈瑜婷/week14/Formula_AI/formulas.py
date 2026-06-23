from fastmcp import FastMCP
import math

mcp = FastMCP(
    name="Formulas-MCP-Server",
    instructions="""This server contains some api of news.""",
)
@mcp.tool
def get_bmr_rate(weight: float, height: float):
    """
    在医疗健康与营养评估领域，基础代谢率（BMR）是衡量个体在静息状态下维持基本生理
功能所需能量的重要指标。它广泛应用于能量需求评估、体重管理、临床营养支持等多个场
景。为了便于快速估算BMR，通常采用经验性公式进行建模。本模型基于线性关系假设，
构建了一个简化的确定性模型，旨在通过个体的体重和身高数据快速估算其每日基础代谢所
需热量。该模型省略了年龄、性别等复杂因素，适用于初步筛查或通用场景的能量需求估算。
    :param weight: 个体体重，单位为千克（kg）；
    :param height: 个体身高，单位为厘米（cm）。
    :return: 基础代谢率（BMR）
    """
    return round(10 * weight + 6.25 * height - 100, 2)


@mcp.tool
def get_dissolved_oxygen(t: float, a: float, b: float, c: float, d: float):
    """
    在水产养殖系统中，溶解氧（Dissolved Oxygen, DO）是影响水生生物健康和生长的关键
环境因子之一。其浓度受多种因素影响，包括水体自净能力、微生物活动、水生生物呼吸作用
以及外界环境（如温度、光照、风力等）的周期性变化。为了更好地理解和预测DO的动态变化
趋势，建立一个能够反映其非线性行为的数学模型具有重要意义。该模型可用于模拟封闭或半
封闭养殖系统中DO浓度随时间演变的过程，为水质调控和管理提供理论支持。
    
    基于非线性动力学的建模思路，构建如下表达式以描述溶解氧浓度随时间变化的规律：
    DO(t) = a·e^(-b·t) + c·sin(d·t)
    
    该模型融合了指数衰减项和周期性扰动项，分别反映溶解氧的自然消耗过程和环境因素引起
的波动特性。通过该模型，可以对养殖系统中溶解氧的动态变化进行定量模拟与分析，为水质
调控策略的制定提供依据。
    
    :param t: 时间，单位为小时（h）或其他时间单位；
    :param a: 初始溶解氧释放量，反映系统初始状态下的氧含量；
    :param b: 溶解氧的衰减系数，刻画其随时间自然下降的速率；
    :param c: 环境扰动的振幅，体现外部周期性因素（如昼夜变化）对DO浓度的影响强度；
    :param d: 环境扰动的频率，反映扰动周期的快慢。
    :return: 溶解氧浓度 DO(t)，单位为mg/L或其他浓度单位
    """
    return round(a * math.exp(-b * t) + c * math.sin(d * t), 4)


@mcp.tool
def predict_daily_orders(ad_spend: float, discount_rate: float, prev_orders: float, 
                        alpha: float = 0.05, beta: float = 100.0, gamma: float = 0.7):
    """
    在电子商务运营中，准确预测每日订单增长量对于库存管理、资源配置和营销策略优化具有
重要意义。为了构建一个简洁且具有解释性的预测模型，我们考虑从三个关键业务驱动因素出发：
广告支出（ad_spend）、当日折扣力度（discount_rate）以及前日订单数量（prev_orders）。
这些变量分别代表了市场推广投入、价格刺激效应以及订单增长的惯性趋势。
    
    基于上述变量，我们设计了一个用于预测当日订单数量的一阶线性差分方程模型。该模型以历史
数据为基础，通过量化各因素对订单增长的影响程度，建立输入变量与输出目标之间的动态关系。
模型具有良好的可解释性和一定的预测能力，适用于短期订单趋势模拟与敏感性分析。
    
    建模公式：
    orders_t = α·ad_spend_t + β·discount_rate_t + γ·prev_orders_t
    
    该模型通过线性组合的方式，综合评估各因素对当日订单增长的贡献，形成对订单数量的预测值。
    
    :param ad_spend: 当日广告支出金额，单位为元或其他货币单位；
    :param discount_rate: 当日折扣力度，通常为0-1之间的小数（如0.2表示8折）；
    :param prev_orders: 前一日订单数量，用于反映订单趋势的惯性影响；
    :param alpha: 广告支出对订单量的敏感系数，默认值为0.05；
    :param beta: 折扣率对订单量的放大系数，默认值为100；
    :param gamma: 前一日订单数量对当前日订单趋势的惯性影响系数，默认值为0.7。
    :return: 预测的当日订单数量
    """
    return round(alpha * ad_spend + beta * discount_rate + gamma * prev_orders, 2)


@mcp.tool
def predict_crop_yield(fertility: float, irrigation: float, temperature: float,
                      a: float = 2.5, b: float = 1.2, c: float = 0.3):
    """
    在农业科研领域，准确预测作物产量对于制定种植策略、优化资源配置以及提升农业生产效率
具有重要意义。本模型旨在构建一个基于关键环境与土壤因素的确定性方程，用于估算单位面积上
的作物产量。该模型综合考虑了土壤肥力、灌溉量以及气温对作物生长的影响，适用于在可控环境
条件下对产量进行定量分析和趋势预测。
    
    建模公式：
    Y = a·F + b·I - c·T²
    
    其中，Y表示单位面积作物产量（kg/ha），F为土壤肥力指数，I为每周灌溉量（mm/week），
T为平均气温（℃）。经验系数a, b, c分别反映各因素对产量的贡献程度。该公式体现了肥力和
灌溉的正向促进作用，以及温度偏离适宜范围后对产量的抑制效应。
    
    :param fertility: 土壤肥力指数F，通常为0-100的无量纲指标；
    :param irrigation: 每周灌溉量I，单位为mm/week；
    :param temperature: 平均气温T，单位为℃；
    :param a: 肥力对产量的贡献系数，默认值为2.5；
    :param b: 灌溉对产量的贡献系数，默认值为1.2；
    :param c: 温度平方项的抑制系数，默认值为0.3。
    :return: 单位面积作物产量Y，单位为kg/ha
    """
    return round(a * fertility + b * irrigation - c * (temperature ** 2), 2)


@mcp.tool
def calculate_food_evaporation(initial_moisture: float, evaporation_rate: float, drying_time: float):
    """
    在食品加工与制造过程中，干燥是一个关键的工艺环节，广泛应用于食品保存、品质控制及延长
货架期等方面。水分含量的控制对于确保食品的稳定性和安全性至关重要。为了有效管理干燥过程，
需要建立能够准确预测食品在干燥过程中水分变化的数学模型。该模型有助于优化干燥时间、温度
以及其他工艺参数，从而提高生产效率并保证产品质量。
    
    本模型聚焦于食品干燥过程中的水分蒸发行为，假设水分蒸发速率与当前水分含量成正比，通过
积分方法计算在给定时间范围内食品的累计水分蒸发量。该模型可作为干燥过程模拟与优化的基础
工具。
    
    建模公式：
    
    模型中水分含量随时间的变化关系表示为：
    M(t) = M₀·e^(-kt)
    
    累计水分蒸发量的表达式为：
    Evaporated(T) = ∫₀ᵀ M₀(1 - e^(-kt)) dt
    
    通过积分计算可得：
    Evaporated(T) = M₀·(T + (e^(-kT) - 1)/k)
    
    该公式可用于预测在干燥时间T内食品的总水分损失量，为食品干燥工艺的设计与控制提供理论
依据。
    
    :param initial_moisture: 初始水分含量M₀，单位可以是克（g）、千克（kg）或百分比（%）；
    :param evaporation_rate: 水分蒸发速率常数k，单位通常为1/小时（1/h）；
    :param drying_time: 干燥时间T，单位为小时（h）。
    :return: 累计水分蒸发量，单位与initial_moisture保持一致
    """
    return round(initial_moisture * (drying_time + (math.exp(-evaporation_rate * drying_time) - 1) / evaporation_rate), 4)


@mcp.tool
def evaluate_student_performance(study_hours: float, attendance_rate: float, quiz_average: float, 
                                participation: float, w1: float = 0.3, w2: float = 0.25, 
                                w3: float = 0.3, w4: float = 0.15, alpha: float = 0.1, 
                                beta: float = 50.0):
    """
    在教育培训领域，评估学生的学习效果是衡量教学质量和课程成效的重要环节。为了更系统地
理解和预测学生在课程中的表现，构建了一个基于关键影响因素的确定性模型。该模型综合考虑了
学习时长、出勤率、平时测验成绩以及课堂参与度四个核心变量，旨在通过量化方式反映学生的学习
成果，并模拟其在学习过程中的非线性增长趋势。
    
    该模型可用于学生表现预测、教学反馈分析以及个性化学习路径优化等场景，为教育决策提供
数据支持。
    
    建模公式：
    Score = 100 / (1 + e^(-α(w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ - β)))
    
    其中：
    x₁ 表示学习时长（小时）
    x₂ 表示出勤率（百分比）
    x₃ 表示平时测验平均分（百分比）
    x₄ 表示课堂参与度（1-5分）
    w₁, w₂, w₃, w₄ 为对应变量的权重系数
    α 控制S型曲线的陡峭程度
    β 控制曲线在横轴上的平移位置
    
    该公式通过加权线性组合构建输入项，并利用Sigmoid型函数将其压缩至[0,100]区间，从而模拟
学习效果的饱和特性与非线性提升趋势。
    
    :param study_hours: 学习时长x₁，单位为小时（h）；
    :param attendance_rate: 出勤率x₂，取值范围为0-100的百分比；
    :param quiz_average: 平时测验平均分x₃，取值范围为0-100的百分比；
    :param participation: 课堂参与度x₄，取值范围为1-5分；
    :param w1: 学习时长的权重系数，默认值为0.3；
    :param w2: 出勤率的权重系数，默认值为0.25；
    :param w3: 平时测验的权重系数，默认值为0.3；
    :param w4: 课堂参与度的权重系数，默认值为0.15；
    :param alpha: S型曲线陡峭程度控制参数，默认值为0.1；
    :param beta: 曲线平移位置控制参数，默认值为50.0。
    :return: 学生表现评分Score，取值范围为0-100
    """
    # 计算加权线性组合
    weighted_sum = w1 * study_hours + w2 * attendance_rate + w3 * quiz_average + w4 * participation
    # 应用Sigmoid函数
    score = 100 / (1 + math.exp(-alpha * (weighted_sum - beta)))
    return round(score, 2)


@mcp.tool
def predict_time_series(x1_t: float, x2_t: float, x3_t: float, y_t1: float, y_t2: float,
                       a: float = 0.5, b: float = 0.3, c: float = 0.1, d: float = 0.2):
    """
    在系统建模与时间序列预测任务中，差分方程是一种常用工具，能够描述系统当前状态与历史
状态之间的动态关系。该模型特别适用于具有时序依赖特性的系统，其中当前输出不仅依赖于当前
输入，还受到前几个时间步状态的影响。本建模任务构建了一个包含三个输入变量的差分方程模型，
用于模拟并预测系统的下一状态。该模型结构简洁，便于分析系统内部变量之间的相互作用，并可用于
多变量输入场景下的动态行为预测。
    
    建模公式：
    y_t = a·x₁,t + b·y_(t-1) + c·y_(t-2) + d·x₂,t·x₃,t
    
    其中：
    y_t 表示当前时刻的输出值
    x₁,t, x₂,t, x₃,t 分别表示当前时刻的三个输入变量
    y_(t-1) 和 y_(t-2) 表示前两个时间步的输出值，作为系统状态的反馈
    a, b, c, d 为模型参数，用于调节各输入项对输出的影响权重
    
    该模型综合考虑了外部输入与内部状态的耦合关系，具有良好的可解释性和灵活性，适用于
多输入动态系统的建模与预测。
    
    :param x1_t: 当前时刻的第一个输入变量x₁,t；
    :param x2_t: 当前时刻的第二个输入变量x₂,t；
    :param x3_t: 当前时刻的第三个输入变量x₃,t；
    :param y_t1: 前一个时间步的输出值y_(t-1)，表示系统的历史状态；
    :param y_t2: 前两个时间步的输出值y_(t-2)，表示系统的更早历史状态；
    :param a: 第一个输入变量x₁,t的权重系数，默认值为0.5；
    :param b: 前一时刻输出y_(t-1)的权重系数，默认值为0.3；
    :param c: 前两时刻输出y_(t-2)的权重系数，默认值为0.1；
    :param d: 输入变量交互项x₂,t·x₃,t的权重系数，默认值为0.2。
    :return: 当前时刻的预测输出值y_t
    """
    # 计算差分方程
    y_t = a * x1_t + b * y_t1 + c * y_t2 + d * x2_t * x3_t
    return round(y_t, 4)


@mcp.tool
def calculate_quadratic_model(x: float):
    """
    在系统行为分析与预测任务中，常常采用确定性模型来描述输入与输出之间的明确关系。该模型
基于一个预定义的数学关系，确保在相同输入条件下始终产生一致的输出结果。此类模型广泛应用于
工程计算、经济预测以及自然科学领域，适用于那些具有明确因果关系的场景。
    
    本模型构建了一个基于二次函数关系的确定性模型，用于演示如何通过函数表达式对输入变量
进行系统性映射，从而获得对应的输出值。该建模方法不仅具备良好的可解释性，而且在计算效率和
实现复杂度上均具有显著优势。
    
    建模公式：
    y = 2x² + 3x + 1
    
    其中，x表示输入变量，y表示对应的输出结果。该公式定义了一个非线性的二次关系，能够反映
输入变化对输出的影响趋势。
    
    :param x: 输入变量，可以是任意实数值。
    :return: 输出结果y，根据二次函数关系计算得到
    """
    return round(2 * (x ** 2) + 3 * x + 1, 4)


@mcp.tool
def calculate_media_influence(content_quality: float, channels: float, engagement: float, time: float):
    """
    在文化传媒领域，评估一个传播项目的综合影响力是制定传播策略和优化资源配置的重要依据。
为了实现这一目标，建立一个基于关键变量的数学模型，有助于量化内容传播过程中各因素的贡献度。
该模型以内容质量、传播渠道数量、受众参与度以及传播持续时间作为核心输入变量，旨在模拟影响力
随时间的累积过程，并提供可计算的综合影响力指标。
    
    建模公式：
    Influence = content_quality · channels · engagement · time
    
    该模型采用一个简化的积分方程形式，描述影响力在传播时间范围内的累积效应。假设传播渠道
数量与受众参与度在时间维度上保持恒定，积分方程可被简化为四个变量的线性乘积关系。最终的
影响力得分反映了内容质量的高低、传播广度的大小、受众活跃程度的强弱以及传播周期的长短对
整体传播效果的联合影响。
    
    该公式适用于对文化传播项目进行量化分析与横向比较。
    
    :param content_quality: 内容质量评分，通常为0-10的标准化评分，反映内容的创作水平和吸引力；
    :param channels: 传播渠道数量，表示使用的传播平台或渠道的总数；
    :param engagement: 受众参与度，通常为0-1的标准化指标，反映受众的互动活跃程度；
    :param time: 传播持续时间，单位可以是天、周或其他时间单位，表示传播活动的持续周期。
    :return: 综合影响力得分Influence，数值越大表示传播影响力越强
    """
    return round(content_quality * channels * engagement * time, 2)


@mcp.tool
def calculate_soil_moisture(depth: float, initial_moisture: float, decay_coefficient: float):
    """
    在农业和环境科学领域，理解土壤水分分布和迁移对于作物生长、灌溉管理以及地下水补给评估
具有重要意义。土壤水分的垂直分布受降水、蒸发、渗透以及土壤物理性质等多种因素的影响。为了
描述土壤水分的垂直变化，通常采用扩散方程进行建模。该模型基于质量守恒原理和达西定律，能够
有效反映土壤中水分的扩散行为。
    
    本模型使用简化的一维扩散方程来描述土壤水分含量与深度的关系，适用于稳态条件下的土壤
水分分布模拟。
    
    建模公式：
    
    基本的一维扩散方程为：
    ∂θ/∂t = D·∂²θ/∂z²
    
    其中，θ(z,t)表示土壤水分含量，D表示水分扩散系数，z表示土壤深度，t表示时间。
    
    在稳态条件下（忽略时间变化），模型可简化为关于深度的指数衰减函数：
    θ(z) = θ₀·e^(-kz)
    
    其中：
    θ₀ 表示初始表层土壤水分含量
    k 表示衰减系数，与土壤性质相关，反映土壤水分随深度下降的速率
    
    该简化模型能够有效描述在非动态扰动条件下土壤水分随深度变化的基本趋势。
    
    :param depth: 土壤深度z，单位通常为米（m）或厘米（cm）；
    :param initial_moisture: 初始表层土壤水分含量θ₀，单位可以是体积百分比（%）或其他含水量单位；
    :param decay_coefficient: 衰减系数k，单位为1/m或1/cm，反映水分随深度减少的速率。
    :return: 指定深度处的土壤水分含量θ(z)，单位与initial_moisture保持一致
    """
    return round(initial_moisture * math.exp(-decay_coefficient * depth), 4)


@mcp.tool
def calculate_cstr_concentration_rate(CA: float, CA0: float, flow_ratio: float, 
                                     pre_exponential: float, activation_energy: float, 
                                     temperature: float, R: float = 8.314):
    """
    在化工过程中，连续搅拌釜式反应器(CSTR)是用于连续化学反应的常见设备。该模型描述了
CSTR中一级放热反应的动态行为，重点关注反应物浓度随时间的变化规律。这种浓度变化受进料浓度
和流量的影响，同时反应速率也通过反应速率常数受到温度的非线性调控。通过建立常微分方程(ODE)
模型，可以对反应过程进行动态模拟和最优控制设计。
    
    建模公式：
    dCA/dt = (v/V)·(CA0 - CA) - k0·e^(-Ea/(R·T))·CA
    
    其中：
    CA 表示反应物A的瞬时浓度，单位为mol/m³
    CA0 表示进料中反应物A的浓度，单位为mol/m³
    k0 表示反应速率常数的指前因子，单位为1/s
    Ea 表示反应活化能，单位为J/mol
    R 表示理想气体常数，值为8.314 J/(mol·K)
    T 表示反应系统的绝对温度，单位为K
    v/V 表示体积流量(v)与反应器体积(V)的比值，单位为1/s
    
    该模型综合考虑了物料平衡项和化学反应动力学项，适用于描述连续操作条件下反应物浓度的
动态演变过程。
    
    :param CA: 反应物A的瞬时浓度，单位为mol/m³；
    :param CA0: 进料中反应物A的浓度，单位为mol/m³；
    :param flow_ratio: 体积流量与反应器体积的比值(v/V)，单位为1/s；
    :param pre_exponential: 反应速率常数的指前因子k0，单位为1/s；
    :param activation_energy: 反应活化能Ea，单位为J/mol；
    :param temperature: 反应系统的绝对温度T，单位为K；
    :param R: 理想气体常数，默认值为8.314 J/(mol·K)。
    :return: 反应物浓度的变化率dCA/dt，单位为mol/(m³·s)
    """
    # 计算物料平衡项
    material_balance = flow_ratio * (CA0 - CA)
    # 计算反应动力学项
    reaction_rate = pre_exponential * math.exp(-activation_energy / (R * temperature)) * CA
    # 计算浓度变化率
    dCA_dt = material_balance - reaction_rate
    return round(dCA_dt, 6)


if __name__ == "__main__":
    mcp.run(transport="sse", port=8900)