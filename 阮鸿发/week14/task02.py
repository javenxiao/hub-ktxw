
import os
import fitz
import math
import torch
import zipfile
import requests
import numpy as np
from tqdm import tqdm
from typing import Union
from sympy import sympify, SympifyError
from sympy.parsing.latex import parse_latex
from fastmcp import FastMCP



# LOCAL_EMBED_MODEL_PATH = "../../../models/google-bert/bert-base-chinese"

# PDF文档纯文本解析
def build_simple_knowledge_base():
    """从文件中解析所有PDF和Markdown文档，提取纯文本。"""
    print("--- 1. 正在构建纯文本知识库 ---")
    knowledge_base = []

    # 直接从 documents 目录读取文件
    documents_dir = './document_10'
    if not os.path.exists(documents_dir):
        print(f"目录 {documents_dir} 不存在")
        return knowledge_base

    # 获取所有 PDF 和 Markdown 文件
    doc_files = []
    for root, dirs, files in os.walk(documents_dir):
        for file in files:
            if file.endswith(('.pdf', '.md')):
                doc_files.append(os.path.join(root, file))

    for doc_path in tqdm(doc_files, desc="解析文档纯文本"):
        doc_id, full_text = os.path.basename(doc_path), ""
        try:
            if doc_path.endswith('.pdf'):
                with fitz.open(doc_path) as doc:
                    for page in doc:
                        full_text += page.get_text() + "\n"
            elif doc_path.endswith('.md'):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            knowledge_base.append({"id": doc_id, "full_text": full_text.strip()})
        except Exception as e:
            print(f"处理文件 {doc_path} 时出错: {e}")
            knowledge_base.append({"id": doc_id, "full_text": ""})
    return knowledge_base

# print (build_simple_knowledge_base())
# import pdb; pdb.set_trace()

# sympy公式解析
def calculate_with_sympy(formula_string: str, params: dict) -> Union[float, None]:
    """使用SymPy对公式和参数进行精确计算。"""
    if not formula_string or not isinstance(params, dict): return None
    try:
        if '\\' in formula_string:
            if '=' in formula_string: formula_string = formula_string.split('=', 1)[1]
            formula_string = formula_string.replace('\\cdot', '*')
            expr = parse_latex(formula_string)
        else:
            if '=' in formula_string:
                formula_string = formula_string.split('=', 1)[1]
            elif '一' in formula_string:
                formula_string = formula_string.split('一', 1)[1]
            formula_string = formula_string.replace('（', '(').replace('）', ')').replace('×', '*').replace('÷',
                                                                                                          '/').replace(
                '－', '-')
            expr = sympify(formula_string)

        safe_params = {}
        for k, v in params.items():
            try:
                safe_params[k] = float(v)
            except (ValueError, TypeError):
                continue

        free_symbols_str = {str(s) for s in expr.free_symbols}
        sympy_params = {k: v for k, v in safe_params.items() if k in free_symbols_str}

        if len(sympy_params) != len(expr.free_symbols):
            return None

        result = expr.subs(sympy_params).evalf()
        return float(result)
    except (SympifyError, SyntaxError, TypeError, Exception, ValueError):
        return None

# numpy数值解析
def calculate_with_numpy(formula_string: str, params: dict) -> Union[float, None]:
    """使用NumPy对公式和参数进行计算。"""
    if not formula_string or not isinstance(params, dict):
        return None

    try:
        # 清理公式字符串
        if '=' in formula_string:
            formula_string = formula_string.split('=', 1)[1]
        elif '一' in formula_string:
            formula_string = formula_string.split('一', 1)[1]

        formula_string = formula_string.replace('（', '(').replace('）', ')').replace('×', '*').replace('÷', '/').replace('－', '-')

        # 安全参数转换
        safe_params = {}
        for k, v in params.items():
            try:
                safe_params[k] = float(v)
            except (ValueError, TypeError):
                continue

        # 创建安全的命名空间
        namespace = {
            'np': np,
            '__builtins__': {},
        }
        namespace.update(safe_params)

        # 计算结果
        result = eval(formula_string, namespace)
        return float(result)
    except Exception as e:
        return None


mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server Comprehension of Document Formula Analysis and Intelligent Question Answering of tools.""",
)

# 定义对应的MCP工具
@mcp.tool
def get_nonlinear_interaction(x: float, y: float):
    """
    在复杂系统分析中，常常需要构建能够反映变量间非线性交互作用的数学模型。该模型旨在模拟两个输入变量 x 和 y 对某一目标输出的综合影响，
    其中包含了周期性变化与线性交互的成分。该建模方法适用于描述如环境因素对系统响应的影响、多因子耦合作用下的信号响应机制等场景。
    尽管模型本身为确定性函数，但其结构设计使得输出呈现出类随机波动的特性，从而更好地模拟真实世界中的复杂行为。
    param: x；
    param: y；
    return: fun
    """
    try:
        return round(2.5 * math.sin(x) + 1.8 * math.cos(y) + 0.3 * x * y)
    except:
        return []


# @mcp.tool
def calculate_dissolved_oxygen(t: float, a: float, b: float, c: float, d: float):
    """计算水产养殖系统中溶解氧浓度随时间的变化
    
    基于非线性动力学的建模思路，构建如下表达式以描述溶解氧浓度随时间变化的规律：
    𝐷𝑂(𝑡) = 𝑎⋅𝑒^(-𝑏⋅𝑡) + 𝑐⋅sin(𝑑⋅𝑡)
    
    参数:
    t: 时间
    a: 初始溶解氧释放量
    b: 溶解氧的衰减系数
    c: 环境扰动的振幅
    d: 环境扰动的频率
    
    返回:
    溶解氧浓度
    """
    try:
        return f"溶解氧浓度= {a * np.exp(-b * t) + c * np.sin(d * t)}"
    except:
        return None


@mcp.tool
def calculate_yield_prediction(temp: float, rainfall: float, fertilizer: float, sunlight: float, soil_quality: float):
    """农业产量预测模型
    
    农业产量受到多种环境和管理因素的综合影响，为了量化这些因素对作物产量的作用，构建了一个简化的代数模型。
    该模型综合考虑了五个关键变量：平均生长温度、生长期间降水量、施肥量、每日平均光照时长以及土壤质量指数。
    
    模型的基本形式为：
    yield_prediction = base_yield × temp_factor × rainfall_factor × fertilizer_factor × sunlight_factor × soil_factor
    
    参数:
    temp: 平均生长温度
    rainfall: 生长期间降水量
    fertilizer: 施肥量
    sunlight: 每日平均光照时长
    soil_quality: 土壤质量指数
    
    返回:
    预测产量
    """
    try:
        base_yield = 5.0  # 基础产量水平，单位吨/公顷
        
        # 温度因子，以25℃为最适温度
        temp_factor = 1.0 - abs(temp - 25) / 25
        
        # 降水量因子，以600mm为最优
        rainfall_factor = 1.0 - abs(rainfall - 600) / 600
        
        # 施肥量因子，与产量正相关
        fertilizer_factor = 1.0 + fertilizer / 200
        
        # 光照时长因子，在8~12小时范围内最为有利
        sunlight_factor = 0.8 + sunlight / 12 * 0.4
        
        # 土壤质量因子，土壤质量越高促进作用越强
        soil_factor = 1.0 + soil_quality
        
        # 计算最终产量
        yield_prediction = base_yield * temp_factor * rainfall_factor * fertilizer_factor * sunlight_factor * soil_factor
        
        # 产量不能为负
        return max(0, yield_prediction)
    except:
        return None


@mcp.tool
def predict_daily_orders(ad_spend: float, discount_rate: float, prev_orders: float):
    """电子商务运营中每日订单增长量预测模型
    
    在电子商务运营中，准确预测每日订单增长量对于库存管理、资源配置和营销策略优化具有重要意义。
    模型考虑从三个关键业务驱动因素出发：广告支出、当日折扣力度以及前一天订单数量。
    
    建模公式:
    orders_t = α * ad_spend_t + β * discount_rate_t + γ * prev_orders_t
    
    其中，α=0.05 表示广告支出对订单量的敏感系数，
    β=100 表示折扣率对订单量的放大系数，
    γ=0.7 表示前一日订单数量对当前日订单趋势的惯性影响。
    
    参数:
    ad_spend: 广告支出
    discount_rate: 当日折扣力度
    prev_orders: 前一天订单数量
    
    返回:
    预测的当日订单数量
    """
    try:
        alpha = 0.05  # 广告支出对订单量的敏感系数
        beta = 100    # 折扣率对订单量的放大系数
        gamma = 0.7   # 前一日订单数量对当前日订单趋势的惯性影响
        
        orders_t = alpha * ad_spend + beta * discount_rate + gamma * prev_orders
        return orders_t
    except:
        return None


@mcp.tool
def calculate_moisture_evaporation(M0: float, k: float, T: float):
    """食品加工与制造过程中水分蒸发量计算模型
    
    在食品加工与制造过程中，干燥是一个关键的工艺环节，广泛应用于食品保存、品质控制及延长货架期等方面。
    该模型聚焦于食品干燥过程中的水分蒸发行为，假设水分蒸发速率与当前水分含量成正比。
    
    建模公式:
    累计水分蒸发量的表达式为：
    Evaporated(T) = M0 * (T + (exp(-k*T) - 1) / k)
    
    其中 M(t) = M0 * exp(-k*t) 表示t时刻食品的水分含量，
    M0 为初始水分含量，
    k 为水分蒸发速率常数，
    t 为干燥时间。
    
    参数:
    M0: 初始水分含量
    k: 水分蒸发速率常数
    T: 干燥时间
    
    返回:
    在干燥时间T内食品的总水分损失量
    """
    try:
        evaporated_T = M0 * (T + (np.exp(-k * T) - 1) / k)
        return evaporated_T
    except:
        return None


@mcp.tool
def predict_crop_yield(F: float, I: float, T: float):
    """农业科研领域作物产量预测模型
    
    在农业科研领域，准确预测作物产量对于制定种植策略、优化资源配置以及提升农业生产效率具有重要意义。
    该模型旨在构建一个基于关键环境与土壤因素的确定性方程，用于估算单位面积上的作物产量。
    模型综合考虑了土壤肥力、灌溉量以及气温对作物生长的影响。
    
    建模公式:
    Y = a * F + b * I - c * T^2
    
    其中，Y 表示单位面积作物产量（kg/ha），
    F 为土壤肥力指数，
    I 为每周灌溉量（mm/week），
    T 为平均气温（℃）。
    经验系数 a, b, c 分别反映各因素对产量的贡献程度，模型中设定为常数。
    
    参数:
    F: 土壤肥力指数
    I: 每周灌溉量（mm/week）
    T: 平均气温（℃）
    
    返回:
    单位面积作物产量（kg/ha）
    """
    try:
        a = 100  # 土壤肥力对产量的贡献系数
        b = 50   # 灌溉量对产量的贡献系数
        c = 0.5  # 气温对产量的抑制系数
        
        Y = a * F + b * I - c * T**2
        return Y
    except:
        return None


@mcp.tool
def evaluate_student_performance(x1: float, x2: float, x3: float, x4: float):
    """教育培训领域学生学习效果评估模型
    
    在教育培训领域，评估学生的学习效果是衡量教学质量和课程成效的重要环节。
    该模型综合考虑了学习时长、出勤率、平时测验成绩以及课堂参与度四个核心变量，
    旨在通过量化方式反映学生的学习成果，并模拟其在学习过程中的非线性增长趋势。
    
    建模公式:
    Score = 100 / (1 + exp(-alpha * (w1*x1 + w2*x2 + w3*x3 + w4*x4 - beta)))
    
    其中：
    x1 表示学习时长（小时）
    x2 表示出勤率（百分比）
    x3 表示平时测验平均分（百分比）
    x4 表示课堂参与度（1~5分），经过线性映射后参与计算
    w1, w2, w3, w4 为对应变量的权重系数
    alpha 控制S型曲线的陡峭程度
    beta 控制曲线在横轴上的平移位置
    
    参数:
    x1: 学习时长（小时）
    x2: 出勤率（百分比）
    x3: 平时测验平均分（百分比）
    x4: 课堂参与度（1~5分）
    
    返回:
    学生学习效果评分（0-100分）
    """
    try:
        # 权重系数
        w1, w2, w3, w4 = 0.3, 0.2, 0.4, 0.1
        alpha = 0.1  # S型曲线的陡峭程度
        beta = 50    # 曲线在横轴上的平移位置
        
        linear_combination = w1*x1 + w2*x2 + w3*x3 + w4*x4
        score = 100 / (1 + np.exp(-alpha * (linear_combination - beta)))
        return score
    except:
        return None


@mcp.tool
def predict_system_state(x1: float, y_prev1: float, y_prev2: float, x2: float, x3: float):
    """系统建模与时间序列预测模型
    
    在系统建模与时间序列预测任务中，差分方程是一种常用工具，能够描述系统当前状态与历史状态之间的动态关系。
    该模型特别适用于具有时序依赖特性的系统，其中当前输出不仅依赖于当前输入，还受到前几个时间步状态的影响。
    本建模任务构建了一个包含三个输入变量的差分方程模型，用于模拟并预测系统的下一状态。
    
    建模公式:
    y_t = a * x1_t + b * y_(t-1) + c * y_(t-2) + d * x2_t * x3_t
    
    其中，y_t 表示当前时刻的输出值；
    x1_t, x2_t, x3_t 分别表示当前时刻的三个输入变量；
    y_(t-1) 和 y_(t-2) 表示前两个时间步的输出值，作为系统状态的反馈；
    a, b, c, d 为模型参数，用于调节各输入项对输出的影响权重。
    
    参数:
    x1: 当前时刻的第一个输入变量
    y_prev1: 前一个时间步的输出值
    y_prev2: 前两个时间步的输出值
    x2: 当前时刻的第二个输入变量
    x3: 当前时刻的第三个输入变量
    
    返回:
    当前时刻的系统输出值
    """
    try:
        # 模型参数
        a, b, c, d = 0.5, 0.3, 0.1, 0.2
        
        y_t = a * x1 + b * y_prev1 + c * y_prev2 + d * x2 * x3
        return y_t
    except:
        return None


@mcp.tool
def calculate_quadratic_function(x: float):
    """确定性二次函数模型
    
    在系统行为分析与预测任务中，常常采用确定性模型来描述输入与输出之间的明确关系。
    该模型基于一个预定义的数学关系，确保在相同输入条件下始终产生一致的输出结果。
    此类模型广泛应用于工程计算、经济预测以及自然科学领域，适用于那些具有明确因果关系的场景。
    
    建模公式:
    y = 2*x^2 + 3*x + 1
    
    其中，x 表示输入变量，y 表示对应的输出结果。
    该公式定义了一个非线性的二次关系，能够反映输入变化对输出的影响趋势。
    
    参数:
    x: 输入变量
    
    返回:
    输出结果
    """
    try:
        y = 2 * x**2 + 3 * x + 1
        return y
    except:
        return None

# print (calculate_dissolved_oxygen(3,3,3,3,3))
