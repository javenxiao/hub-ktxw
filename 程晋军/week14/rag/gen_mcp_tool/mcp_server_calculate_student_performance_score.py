from fastmcp import FastMCP
import math

# 创建 MCP 服务器实例
mcp = FastMCP("Formula Calculation Server")

@mcp.tool
def calculate_student_performance_score(
    study_hours: float,
    attendance_rate: float,
    quiz_average: float,
    class_participation: float,
    weight_study: float = 0.3,
    weight_attendance: float = 0.2,
    weight_quiz: float = 0.3,
    weight_participation: float = 0.2,
    alpha: float = 0.1,
    beta: float = 5.0
) -> float:
    """
    计算学生综合学习表现得分
    
    该模型基于学习时长、出勤率、平时测验成绩和课堂参与度四个核心变量，通过加权线性组合与Sigmoid函数映射，生成一个介于0到100之间的综合表现评分。
    模型模拟了学习效果的非线性增长特性，即初期提升较快，后期趋于饱和，符合教育心理学中的学习曲线规律。
    
    公式: 
    $$
    \mathrm{Score} = \frac{100}{1 + e^{-\alpha (w_1 x_1 + w_2 x_2 + w_3 x_3 + w_4 x_4 - \beta)}}
    $$
    
    其中：
    - $x_1$: 学习时长（小时）
    - $x_2$: 出勤率（百分比，如85表示85%）
    - $x_3$: 平时测验平均分（百分比）
    - $x_4$: 课堂参与度（1~5分），将按比例映射为0~100区间后参与计算
    - $w_1, w_2, w_3, w_4$: 各变量对应的权重系数，默认分别为0.3、0.2、0.3、0.2
    - $\alpha$: 控制S型曲线陡峭程度，值越大曲线越陡
    - $\beta$: 控制曲线在横轴上的平移位置，影响得分达到50%时的临界点
    
    Args:
        study_hours: 学习时长（单位：小时）
        attendance_rate: 出勤率（百分比，取值范围0-100）
        quiz_average: 平时测验平均分（百分比，取值范围0-100）
        class_participation: 课堂参与度评分（1-5分）
        weight_study: 学习时长的权重系数，默认0.3
        weight_attendance: 出勤率的权重系数，默认0.2
        weight_quiz: 测验成绩的权重系数，默认0.3
        weight_participation: 课堂参与度的权重系数，默认0.2
        alpha: Sigmoid函数的陡峭参数，控制曲线变化速率，默认0.1
        beta: Sigmoid函数的偏移参数，控制曲线中心位置，默认5.0
    
    Returns:
        float: 综合学习表现得分，范围为0.0到100.0
    """
    # 将课堂参与度从1-5分线性映射到0-100分
    participation_scaled = (class_participation - 1) / 4 * 100
    
    # 构建加权线性组合项
    weighted_sum = (
        weight_study * study_hours +
        weight_attendance * attendance_rate +
        weight_quiz * quiz_average +
        weight_participation * participation_scaled
    )
    
    # 应用Sigmoid函数进行非线性压缩
    exponent = alpha * (weighted_sum - beta)
    sigmoid_value = 1 / (1 + pow(2.71828, -exponent))
    
    # 映射到0-100区间
    score = 100 * sigmoid_value
    
    return score

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run(transport="http", port=8002)
