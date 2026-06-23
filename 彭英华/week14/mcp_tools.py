import asyncio
import numpy as np
from scipy import integrate
import sympy as sp
from fastmcp import FastMCP,Client
mcp = FastMCP("My MCP Server")
@mcp.tool
def calculate_agricultural_product(Input_kg:float,Base_price:float,Fluctuation:float):
    """该公式用于模拟农产品零售销售额，考虑了进货量、基准价格以及市场价格的随机波动。通过引入一个在固定区间内均匀分布的随机变量来反映市场不确定性，从而更真实地刻画零售收入随进货量变化的情况。
    params:
       Input_kg:农产品的进货量
       Base_price: 设定的基准价格
       Fluctuation: 表示市场价格波动幅度的随机变量，在固定区间内服从均匀分布
    """
    Revenue = Input_kg*(Base_price+Fluctuation)
    return np.round(Revenue,3)
@mcp.tool
def calculate_Monthly_rental_income(Area:float,Location_Score:float,Age:int,Bedrooms:int,Distance_to_Subway:float):
    """该公式用于估算房产的预期月租金收益，综合考虑了房产面积、地段评分、房龄、卧室数量以及到地铁站的距离等因素。通过将各因素以线性和非线性方式组合，对基础租金进行加权调整，适用于房地产投资与资产管理中的初步租金预测和快速评估。
    params:
     "Area": "房产面积（单位：平方米或其他统一面积单位）",
     "Location_Score": "地段评分（假设为0-10分制，反映地理位置优劣）",
     "Age": "房龄（单位：年）",
     "Bedrooms": "卧室数量（整数）",
     "Distance_to_Subway": "房产到最近地铁站的距离（单位：米）"
    """
    Monthly_Rent = 50*Area*(Location_Score/10)*(1-Age/30)*(1 + 0.1*Bedrooms) * (1000 / (1000 + Distance_to_Subway))
    return np.round(Monthly_Rent,3)
@mcp.tool
def calculate_x_y(x:float,y:float):
    """该公式用于模拟两个输入变量x和y对某一目标输出的综合影响，融合了x和y各自的非线性主效应（通过正弦和余弦函数体现周期性变化）以及它们的线性交互效应（通过乘积项体现）。适用于描述多因子耦合作用下的复杂系统响应，如环境因素对系统输出的影响等场景。
    params:
       "x": "第一个输入变量，影响输出的非线性主效应和交互效应",
       "y": "第二个输入变量，影响输出的非线性主效应和交互效应"
    """
    z = 2.5*np.sin(x)+1.8*np.cos(y)+0.3*x*y
    return np.round(z,3)
@mcp.tool
def calculate_daily_milk(feed_quality:float,health_status:float,avg_temp:float,milk_freq:int,lactation_week:int):
    """该公式用于预测奶牛的日均产奶量，基于确定性建模方法，综合考虑饲料质量、健康状况、环境温度、挤奶频率和泌乳周期五个关键影响因素。以25kg为基础产奶量，通过各因子的乘积逐层调整，反映不同饲养和生理条件下奶牛的产奶潜力。
    params:
        "feed_quality": "饲料质量评分，范围通常为0-100，反映营养供给水平对产奶能力的支持程度",
        "health_status": "健康状态评分，范围通常为0-100，体现奶牛生理状况对产奶表现的影响",
        "avg_temp": "环境平均温度（单位：摄氏度），用于计算温度偏离最适值（20℃）带来的产奶抑制效应",
        "milk_freq": "每日挤奶次数，反映挤奶频率对产奶潜力的正向激励作用",
        "lactation_week": "当前泌乳周数，用于刻画奶牛在泌乳期内产奶量随时间变化的趋势"
    """
    milk_production = 25 * (feed_quality / 100) * (health_status / 100) * (1 - 0.05 * np.abs(avg_temp - 20)) * (milk_freq / 2) * (1 - np.exp(-0.1 * lactation_week))
    return np.round(milk_production,3)
@mcp.tool
def calculate_many_system(x1_val:float,x2_val:float,x3_val:float,x4_val:float,x5_val:float,a_val=2.0,b_val=1.5,c_val=0.5,d_val=0.3,e_val=0.8):
    """该公式用于描述一个具有多变量输入的复杂系统，其输出由多种非线性机制共同决定。具体包括：一个从0到x₁的积分项，用于刻画具有指数衰减特性的累积过程；一个关于x₂的二次项，表示二次响应；一个关于x₃的正弦函数，体现周期性响应；一个对x₄进行对数变换的项（加1以避免对零取对数）；以及一个对x₅的平方根项，代表幂律关系。整体模型通过参数a、b、c、d、e调节各组成部分的强度和动态范围，适用于物理、工程或金融等领域中复杂非线性现象的建模。
    params:
        "a_val": "控制积分项中指数衰减部分的幅值",
        "b_val": "控制积分项中指数衰减速率的时间常数",
        "c_val": "积分项分母中的偏移常数，防止除零并调节衰减曲线形状",
        "d_val": "二次响应项的系数，调节x₂²对输出的影响强度",
        "e_val": "周期性响应项的振幅系数，调节sin(x₃)对输出的贡献"
        "x1_val":"用户输入的变量“
        "x2_val":"用户输入的变量“
        "x3_val":"用户输入的变量“
        "x4_val":"用户输入的变量“
        "x5_val":"用户输入的变量“
    """
    x1, x2, x3, x4, x5, t = sp.symbols('x1 x2 x3 x4 x5 t', real=True, nonnegative=True)
    a, b, c, d, e = sp.symbols('a b c d e', real=True)
    integrand = (a * sp.exp(-t / b)) / (c + t)
    integral_term = sp.Integral(integrand, (t, 0, x1))
    other_terms = d * (x2 ** 2) + e * sp.sin(x3) + sp.log(x4 + 1) + sp.sqrt(x5)
    full_expression = integral_term + other_terms
    try:
        numeric_func = sp.lambdify((x1, x2, x3, x4, x5, a, b, c, d, e),
                                   full_expression,'numpy')
        result = numeric_func(x1_val,x2_val,x3_val,x4_val,x5_val,a_val,b_val,c_val,d_val,e_val)
        return np.round(result)
    except Exception as e:
        try:
            def integrand_func(t_vals):
                return (a_val * np.exp(-t_vals / b_val)) / (c_val + t_vals)

            # 计算数值积分
            integral_val, error = integrate.quad(integrand_func, 0, x1_val)
            other_val = (d_val * (x2_val ** 2) +
                     e_val * np.sin(x3_val) +
                     np.log(x4_val + 1) +
                     np.sqrt(x5_val))
            return np.round(integral_val + other_val,3)
        except Exception as e:
            return None
@mcp.tool
def calculate_weight_trend(W_t:float,C_t:float,E_t:float,k:float):
    """该公式用于模拟和预测个体在一定时间内的体重变化趋势，基于每日热量摄入与消耗之间的平衡。体重的变化量由当天的热量盈亏（摄入减去消耗）除以热量与体重之间的转换系数决定，从而提供一个定量分析工具来辅助制定饮食与运动计划，实现减重或增重目标。
    params:
        "W_t": "第 t 天的体重",
        "C_t": "第 t 天的热量摄入量",
        "E_t": "第 t 天的热量消耗量",
        "k": "热量与体重之间的转换系数"
    """
    result = W_t + (C_t-E_t) / k
    return np.round(result,3)
@mcp.tool
def calculate_financial_price(S:float,K:float,T:int,r:float):
    """该公式是一种简化的金融衍生品定价模型，用于模拟基于偏微分方程（PDE）的期权定价过程。它通过将标的资产价格和行权价格分别进行指数贴现，并取其差值与零的最大值，确保价格非负，从而反映类似欧式看涨期权的收益结构。尽管形式简化，但体现了时间衰减、无风险贴现和资产价格动态等核心定价要素。
    params:
        "S": "标的资产当前价格",
        "K": "行权价格",
        "T": "到期时间（以年为单位）",
        "r": "无风险利率"
    """
    inter = S * np.exp(-r * T) - K * np.exp(-r * T)
    price = np.max(inter,0)
    if price == 0:
        return 0
    else:
        return np.round(price,3)
@mcp.tool
def calculate_ADG(feed_intake:float,protein_content:float,animal_weight:float):
    """该公式用于预测牲畜的日增重（ADG），基于日采食量、饲料粗蛋白含量和动物当前体重三个关键变量。模型假设日增重与日采食量和粗蛋白含量呈正相关，与动物体重呈负相关，体现了随着体重增加，增重效率逐渐降低的生物学规律。此模型可用于不同饲养条件下牲畜生长性能的初步预测与比较分析。
    params:
        "feed_intake": "日采食量，表示动物每日摄入的饲料量",
        "protein_content": "饲料粗蛋白含量，反映饲料中蛋白质的比例",
        "animal_weight": "动物当前体重，作为生长阶段的生理指标"
    """
    ADG = (feed_intake * protein_content) / (animal_weight * 10)
    return np.round(ADG,3)
@mcp.tool
def calculate_dissolved_oxygen(a:float,b:float,c:float,d:float,t:float):
    """该公式用于描述水产养殖系统中溶解氧（DO）浓度随时间变化的动态规律，结合了指数衰减项和周期性扰动项，分别反映溶解氧的自然消耗过程和由环境因素（如昼夜变化）引起的周期性波动。模型可用于模拟封闭或半封闭养殖系统中DO浓度的非线性变化，为水质调控和管理提供理论支持。
    params:
        "a": "初始溶解氧释放量，反映系统初始状态下的氧含量",
        "b": "溶解氧的衰减系数，刻画其随时间自然下降的速率",
        "c": "环境扰动的振幅，体现外部周期性因素（如昼夜变化）对DO浓度的影响强度",
        "d": "环境扰动的频率，反映扰动周期的快慢",
        "t": "时间变量"
    """
    DO = a * np.exp(-b * t) + c * np.sin(d * t)
    return np.round(DO,3)
@mcp.tool
def calculate_order_growth(ad_spend_t:float,discount_rate_t:float,prev_orders_t:float,α=0.05,β=100,γ=0.7):
    """该公式用于预测电子商务中当日的订单数量，基于三个关键业务驱动因素：广告支出、当日折扣力度和前一日的订单数量。模型通过线性组合的方式，量化各因素对订单增长的影响，适用于短期订单趋势预测、库存管理、资源配置及营销策略优化。
    params:
        "α": "广告支出对订单量的敏感系数，值为0.05",
        "ad_spend_t": "当日广告支出",
        "β": "折扣率对订单量的放大系数，值为100",
        "discount_rate_t": "当日折扣力度",
        "γ": "前一日订单数量对当前日订单趋势的惯性影响系数，值为0.7",
        "prev_orders_t": "前一日订单数量"
    """
    orders_t = α * ad_spend_t + β * discount_rate_t + γ * prev_orders_t
    return np.round(orders_t,3)
async def filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available Tool:",[t for t in tools])

if __name__ == "__main__":
    asyncio.run(filtering())
    mcp.run(transport="sse",port=8900)
