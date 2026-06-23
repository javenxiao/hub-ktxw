from fastmcp import FastMCP
import math

# 创建 MCP 服务器实例
mcp = FastMCP("Formula Calculation Server")

@mcp.tool
def calculate_differential_equation(
    current_input1: float,
    current_input2: float,
    current_input3: float,
    previous_output1: float,
    previous_output2: float,
    coefficient_a: float,
    coefficient_b: float,
    coefficient_c: float,
    coefficient_d: float
) -> float:
    """
    基于三输入差分方程的系统状态预测模型
    
    该模型用于预测系统在当前时刻的输出值，综合考虑了当前外部输入与历史状态反馈的影响。适用于具有时序依赖特性的多变量动态系统建模与预测。
    
    公式: 
    $$
    y_t = a \cdot x_{1,t} + b \cdot y_{t-1} + c \cdot y_{t-2} + d \cdot x_{2,t} \cdot x_{3,t}
    $$
    
    其中：
    - $ y_t $：当前时刻的输出值（返回结果）
    - $ x_{1,t}, x_{2,t}, x_{3,t} $：当前时刻的三个输入变量
    - $ y_{t-1}, y_{t-2} $：前一时刻和前两时刻的输出值，作为系统状态的历史反馈
    - $ a, b, c, d $：模型参数，分别控制各输入项对输出的权重影响
    
    Args:
        current_input1: 当前时刻的第一个输入变量 $ x_{1,t} $
        current_input2: 当前时刻的第二个输入变量 $ x_{2,t} $
        current_input3: 当前时刻的第三个输入变量 $ x_{3,t} $
        previous_output1: 前一时刻的输出值 $ y_{t-1} $
        previous_output2: 前两时刻的输出值 $ y_{t-2} $
        coefficient_a: 参数 $ a $，调节 $ x_{1,t} $ 对输出的影响权重
        coefficient_b: 参数 $ b $，调节 $ y_{t-1} $ 的反馈影响权重
        coefficient_c: 参数 $ c $，调节 $ y_{t-2} $ 的反馈影响权重
        coefficient_d: 参数 $ d $，调节 $ x_{2,t} \cdot x_{3,t} $ 乘积项的影响权重
    
    Returns:
        float: 当前时刻的预测输出值 $ y_t $
    """
    result = (
        coefficient_a * current_input1 +
        coefficient_b * previous_output1 +
        coefficient_c * previous_output2 +
        coefficient_d * current_input2 * current_input3
    )
    return result

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run(transport="http", port=8002)
