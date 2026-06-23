from fastmcp import FastMCP
from tool_registry import ToolRegistry # 从tool_registry导入工具注册表
from functools import wraps

# 创建MCP实例
mcp = FastMCP(
    name="Pdf_Formulas",
    instructions="""This server contains formulas of pdfs.""",
)

# 创建全局工具注册表实例
tool_registry = ToolRegistry()


# 增强装饰器：同时装饰为MCP工具并自动注册
def mcp_tool_with_registry(func):
    """增强装饰器：同时将函数装饰为MCP工具并自动注册到工具注册表"""
    # 先用MCP装饰器装饰
    @mcp.tool
    @wraps(func)
    def mcp_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # 再用工具注册表装饰器注册
    registered_wrapper = tool_registry.register_tool(mcp_wrapper)

    return registered_wrapper


# ========== 工具定义 ==========

@mcp_tool_with_registry
def quadratic_equation_with_one_variable(x: float) -> float:
    """在系统行为分析与预测任务中，常常采用确定性模型来描述输入与输出之间的明确关系。该
模型基于一个预定义的数学关系，确保在相同输入条件下始终产生一致的输出结果。此类模
型广泛应用于工程计算、经济预测以及自然科学领域，适用于那些具有明确因果关系的场景
。
本例中，构建了一个基于二次函数关系的确定性模型，用于演示如何通过函数表达式对输入
变量进行系统性映射，从而获得对应的输出值。该建模方法不仅具备良好的可解释性，而且
在计算效率和实现复杂度上均具有显著优势。
: param x：二元一次方程自变量"""
    return 2 * x ** 2 + 3 * x + 1


@mcp_tool_with_registry
def get_the_accumulation_of_influence(content_quality: float, channels: int, engagement: float, time: int) -> float:
    """在文化传媒领域，评估一个传播项目的综合影响力是制定传播策略和优化资源配置的重要依
据。为了实现这一目标，建立一个基于关键变量的数学模型，有助于量化内容传播过程中各
因素的贡献度。该模型以内容质量、传播渠道数量、受众参与度以及传播持续时间作为核心
输入变量，旨在模拟影响力随时间的累积过程，并提供可计算的综合影响力指标。
建模公式
模型采用一个简化的积分方程形式，描述影响力在传播时间范围内的累积效应。假设传播渠
道数量与受众参与度在时间维度上保持恒定，积分方程可被简化为四个变量的线性乘积关系
。最终的影响力得分反映了内容质量的高低、传播广度的大小、受众活跃程度的强弱以及传
播周期的长短对整体传播效果的联合影响。
: param content_quality：内容质量
: param channels：传播广度
: param engagement：受众活跃程度
: param time：传播周期长短
"""
    return content_quality * channels * engagement * time


@mcp_tool_with_registry
def get_forecast_the_trend_in_cattle_herd_size(N_t: float, r: float, K: float) -> float:
    """建模背景
在畜牧业管理中，理解与预测牛群数量的动态变化具有重要意义。为了反映种群在有限资源
环境下的增长特性，采用逻辑斯蒂增长模型的思想，构建一个一阶非线性差分方程模型。该
模型不仅考虑了牛群的自然增长率，还引入了环境承载能力的限制因素，从而更真实地反映
实际种群增长过程。该模型适用于对中长期牛群数量趋势进行预测，辅助制定合理的养殖策
略与资源分配方案。
建模公式
牛群数量随时间演化的差分方程为：
: param N_t：表示第 t 年的牛群数量，
: param r :为年增长率
: param K :为环境承载能力"""
    return N_t + r * N_t * (1 - N_t / K)

@mcp_tool_with_registry
def estimate_property_value(area,floor,age):
    """在房地产市场中，房产的市场价值受多种因素影响，包括面积、楼层和房龄等。这些变量通
常与房屋的使用价值和市场需求密切相关。面积直接决定了房屋的空间大小，是价值评估的
基础；楼层影响居住舒适度和视野，通常对价格产生正向作用；而房产年龄则反映了其新旧
程度和折旧水平，往往与市场价值呈负相关。基于上述逻辑，构建一个经验型评估模型，用
于估算房产的市场价值，有助于辅助定价、投资决策和市场分析。
: param area: 房产面积大小
: param floor: 楼层
: param age: 楼龄
"""
    return 10000 * area * (1 + 0.02 * floor) * (1 - 0.015 * age)

@mcp_tool_with_registry
def predict_texture_hardness(temp,time,sugar):
    """在食品加工与制造过程中，成品的质地硬度是一个关键的质量控制指标，直接影响产品的口
感、储存稳定性及消费者接受度。为了更好地理解和预测加工参数对质地硬度的影响，建立
了一个基于关键工艺参数的线性预测模型。该模型综合考虑了加工温度、加工时间、原料p
H值以及糖分含量四个主要因素，旨在为食品加工过程中的质地控制提供数据支持与优化指
导。
: param temp: 加工温度（单位：℃）
: param time: 加工时间（单位：分钟）
: param ph: 原料pH值（无量纲）
: param sugar: 糖分含量（单位：g/100g）
"""
    return 0.5 * temp + 1.2 * time - 3.0 * ph + 0.8 * sugar

@mcp_tool_with_registry
def predict_combustion_efficiency(hv):
    """在能源化工领域，燃烧过程的效率是评估燃料性能和系统优化的关键指标之一。燃烧效率受
到多种因素的影响，其中燃料的热值（Heating Value, HV）是一个核心
变量。为了建立热值与燃烧效率之间的定量关系，本文构建了一个基于经验数据的确定性模
型。该模型适用于热值范围在40至50
MJ/kg之间的燃料，旨在为燃烧系统的设计与运行提供理论支持和数值预测。
: param hv 热值范围(40至50MJ/kg)
"""
    return 0.85 + 0.005 * (hv - 40)

@mcp_tool_with_registry
def simulate_lake_pollutant_dynamics(q_in,c_in,c,v,k):
    """在环境工程与生态建模中，理解与预测水体中污染物浓度的变化趋势是评估水质状况和制定
污染控制策略的关键环节。湖泊作为重要的地表水体，其污染物浓度受到外部输入与内部过
程的共同影响。本模型旨在描述一个简化但具有实际应用价值的湖泊污染物动态过程，通过
建立常微分方程（ODE）来刻画污染物浓度随时间的变化率。
模型主要考虑两个主导机制：一是外部污染源通过进水持续输入湖泊，二是湖泊内部污染物
因自然过程发生的降解作用。通过该模型，可以对湖泊水质的长期演化趋势进行定量分析，
并为环境管理提供科学依据。
: param c：湖泊中当前污染物浓度（单位：mg/L）
: param q_in：进水流量（单位：m³/天）
: param c_in：进水污染物浓度（单位：mg/L）
: param v：湖泊总体积（单位：m³）
: param k：污染物自然降解速率（单位：1/天）
"""
    return q_in*(c_in - c)/v - k * c

@mcp_tool_with_registry
def calculate_environmental_quality_index(pollution_level,population_density,green_coverage):
    """环境质量指数（Environmental Quality Index, EQI）
是一种综合评估特定区域环境健康状况的指标。该模型旨在通过定量分析污染水平、绿化覆
盖率以及人口密度三者之间的关系，反映城市或区域的环境承载能力和宜居水平。随着城市
化进程的加快，人口集聚与资源消耗加剧了环境压力，因此，构建一个能够反映多因素交互
作用的环境质量评估工具，对于城市规划、生态保护和公共政策制定具有重要意义。
本模型通过引入污染水平作为环境退化的表征，绿化覆盖率作为生态修复能力的指标，以及
人口密度作为人类活动对环境影响的调节因子，构建了一个非线性的确定性评估框架。该框
架能够有效体现环境质量在不同空间尺度上的差异性，为决策者提供科学依据。
: param pollution_level: 污染水平
: param population_density: 人口密度
: param green_coverage: 绿化覆盖面积
"""
    return (100 - pollution_level)/(1+population_density * (1-green_coverage))

@mcp_tool_with_registry
def predict_resting_heart_rate(rhr,age,daily_steps):
    """在医疗健康评估中，静息心率（Resting Heart Rate, RHR）是反
映个体心血管系统健康状况的重要指标之一。研究表明，个体的静息心率受多种因素影响，
其中年龄和日常身体活动水平是两个关键的可控与不可控因素。随着年龄的增长，静息心率
通常会有所上升，而较高的日常活动量则有助于降低静息心率，反映出更好的心肺适应能力
。
为了在临床或健康管理场景中快速评估个体的静息心率水平，构建一个基于关键变量的预测
模型具有重要意义。该模型可用于初步筛查高风险人群、辅助制定个性化干预方案，或作为
健康监测系统中的一个参考模块。
本研究构建了一个线性回归模型，用于预测成年人的静息心率，输入变量包括年龄（age
）和每日平均步数（daily_steps）。模型基于合理假设与模拟数据构建，适用
于初步评估目的。
: param age: 表示个体的年龄（单位：岁）；
: param daily_steps: 表示个体每日平均步数（单位：千步）"""
    return 72 + 0.5 * age - 0.08 * daily_steps
