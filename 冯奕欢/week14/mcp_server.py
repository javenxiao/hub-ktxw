import asyncio
import random
from typing import Annotated

from fastmcp import FastMCP, Client

import sympy as sp
from sympy import exp, sin
from sympy import symbols
from sympy import symbols, sin, cos
from sympy import symbols, Eq, solve

mcp = FastMCP(
    name="MCP Server"
)


@mcp.tool
def student_performance(
    x1: Annotated[float, "学习时长（小时）"],
    x2: Annotated[float, "出勤率（百分比）"],
    x3: Annotated[float, "平时测验平均分（百分比）"],
    x4: Annotated[float, "课堂参与度（1~5分）"],
    w1: Annotated[float, "学习时长的权重系数"],
    w2: Annotated[float, "出勤率的权重系数"],
    w3: Annotated[float, "平时测验平均分的权重系数"],
    w4: Annotated[float, "课堂参与度的权重系数"],
    alpha: Annotated[float, "控制S型曲线的陡峭程度"],
    beta: Annotated[float, "控制曲线在横轴上的平移位置"]
):
    """评估学生的学习效果"""

    # 定义符号变量
    x1_sym, x2_sym, x3_sym, x4_sym = sp.symbols('x1 x2 x3 x4')
    w1_sym, w2_sym, w3_sym, w4_sym = sp.symbols('w1 w2 w3 w4')
    alpha_sym, beta_sym = sp.symbols('alpha beta')

    # 构建公式
    linear_combination = w1_sym * x1_sym + w2_sym * x2_sym + w3_sym * x3_sym + w4_sym * x4_sym - beta_sym
    sigmoid_function = 1 / (1 + sp.exp(-alpha_sym * linear_combination))
    formula = 100 * sigmoid_function

    # 代入参数
    result = formula.subs({
        x1_sym: x1,
        x2_sym: x2,
        x3_sym: x3,
        x4_sym: x4,
        w1_sym: w1,
        w2_sym: w2,
        w3_sym: w3,
        w4_sym: w4,
        alpha_sym: alpha,
        beta_sym: beta
    })

    return float(result)


@mcp.tool
def dissolved_oxygen(
    a: Annotated[float, "初始溶解氧释放量"],
    b: Annotated[float, "溶解氧的衰减系数"],
    c: Annotated[float, "环境扰动的振幅"],
    d: Annotated[float, "环境扰动的频率"],
    t: Annotated[float, "时间"]
):
    """计算溶解氧浓度随时间的变化"""
    from sympy import symbols, exp, sin

    # 定义符号变量
    A, B, C, D, T = symbols('A B C D T')

    # 构建公式
    DO = A * exp(-B * T) + C * sin(D * T)

    # 代入参数值
    result = DO.subs({A: a, B: b, C: c, D: d, T: t})

    return result


@mcp.tool
def tun(
    x: Annotated[float, "第一个输入变量"],
    y: Annotated[float, "第二个输入变量"]
):
    """根据给定的x和y值，计算非线性交互作用模型的结果"""

    # 定义符号变量
    x_sym, y_sym = symbols('x_sym y_sym')

    # 构建公式
    formula = 2.5 * sin(x_sym) + 1.8 * cos(y_sym) + 0.3 * x_sym * y_sym

    # 替换参数并计算结果
    result = formula.subs({x_sym: x, y_sym: y})
    return float(result)


@mcp.tool
def predict_daily_weight_gain(
    x: Annotated[float, "每日饲料摄入量 (kg)"]
):
    """根据饲料摄入量预测牲畜的日增重"""
    from sympy import symbols, simplify

    # 定义符号变量
    X = symbols('X')

    # 构建公式
    formula = 0.5 + 0.3 * X - 0.05 * X**2

    # 计算结果
    result = formula.subs(X, x)
    return float(result)


@mcp.tool
def logistic_growth_model(
    N_t: Annotated[float, "第t年的牛群数量"],
    r: Annotated[float, "年增长率"],
    K: Annotated[float, "环境承载能力"],
    t: Annotated[int, "预测的年数"]
):
    """根据逻辑斯蒂增长模型预测未来t年的牛群数量"""

    # 定义符号变量
    N, R, K_sym, T = symbols('N R K_sym T')

    # 构建差分方程
    equation = Eq(N, N_t + r * N_t * (1 - N_t / K))

    # 初始化牛群数量
    N_t_future = N_t

    # 迭代计算未来t年的牛群数量
    for _ in range(t):
        N_t_future = equation.rhs.subs({N_t: N_t_future, r: r, K: K})

    return N_t_future


@mcp.tool
def loan_balance_update(
    B_t_minus_1: Annotated[float, "上一期的贷款余额"],
    P_t: Annotated[float, "本期客户偿还的本金金额"]
):
    """根据差分方程B_{t}=B_{t-1}-P_{t}计算当前期的贷款余额"""
    B_t, B_t_minus_1_sym, P_t_sym = sp.symbols('B_t B_t_minus_1_sym P_t_sym')
    formula = sp.Eq(B_t, B_t_minus_1_sym - P_t_sym)
    result = formula.subs({B_t_minus_1_sym: B_t_minus_1, P_t_sym: P_t})
    return float(result)


@mcp.tool
def calculate_limit_bending_moment(
    fc: Annotated[float, "混凝土抗压强度 (MPa)"],
    As: Annotated[float, "纵向受拉钢筋面积 (mm^2)"],
    d: Annotated[float, "梁的有效高度 (mm)"]
):
    """计算钢筋混凝土梁的极限抗弯承载力"""

    # 定义符号变量
    fc_sym, As_sym, d_sym = sp.symbols('fc As d')
    fy, b = 400, 300  # 常量参数

    # 构建公式
    Mu = 0.85 * fc_sym * As_sym * d_sym * (1 - (0.59 * As_sym * fy) / (b * d_sym * fc_sym))

    # 代入参数计算结果
    Mu_value = Mu.subs({fc_sym: fc, As_sym: As, d_sym: d})

    return Mu_value


@mcp.tool
def model_with_noise(
    x: Annotated[float, "输入变量x"],
    y: Annotated[float, "输入变量y"]
):
    """模拟一个带有随机噪声的确定性函数"""
    x1, y1 = sp.symbols('x y')
    epsilon = random.uniform(-1, 1)
    formula = x1**2 + y1 + epsilon
    result = formula.subs({x1: x, y1: y})
    return float(result)


@mcp.tool
def linear_model(
        x: Annotated[float, "输入变量"],
        w: Annotated[float, "权重系数"],
        b: Annotated[float, "偏置项"]
):
    """单变量线性模型，用于计算给定输入x通过权重w和偏置b后的输出y"""

    # 定义符号变量
    x_sym, w_sym, b_sym = sp.symbols('x_sym w_sym b_sym')

    # 构建公式
    y_sym = w_sym * x_sym + b_sym

    # 代入参数计算结果
    y = y_sym.subs({x_sym: x, w_sym: w, b_sym: b})

    return float(y)


@mcp.tool
def calculate_adg(
    temp: Annotated[float, "水温（单位：℃）"],
    do: Annotated[float, "溶解氧浓度（单位：mg/L）"],
    ph: Annotated[float, "水体pH值"],
    feed_rate: Annotated[float, "饲料投喂率（单位：%鱼体重）"],
    stock_density: Annotated[float, "养殖密度（单位：kg/m³）"]
):
    """计算鱼类的平均日增重（ADG, Average Daily Gain）"""

    # 定义符号变量
    t, d, p, f, s = sp.symbols('t d p f s')

    # 构建公式
    adg_formula = 0.5 * t + 0.8 * d + 0.3 * p + 1.2 * f - 0.4 * s

    # 替换参数
    adg_value = adg_formula.subs({t: temp, d: do, p: ph, f: feed_rate, s: stock_density})

    return float(adg_value)


async def show_all_mcp():
    """显示所有支持的MCP工具方法"""
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("all tool: ", [tool.name for tool in tools])


if __name__ == '__main__':
    asyncio.run(show_all_mcp())
    mcp.run(transport="sse", port=8900)