from fastmcp import FastMCP
import math

# 创建 MCP 服务器实例
mcp = FastMCP("Formula Calculation Server")

@mcp.tool
def calculate_cumulative_water_evaporation(initial_moisture: float, evaporation_rate_constant: float, drying_time: float) -> float:
    """
    计算食品在干燥过程中累计的水分蒸发量
    
    本模型基于指数衰减原理，假设食品的水分含量随时间呈指数下降，其蒸发速率与当前水分含量成正比。通过积分计算，在给定干燥时间内食品的总水分损失量。
    
    公式: 
    Evaporated(T) = M₀ × (T + (e^(-kT) - 1) / k)
    
    其中：
    - M₀：初始水分含量（单位：kg 或 g）
    - k：水分蒸发速率常数（单位：1/时间，如 1/h）
    - T：干燥时间（单位：小时或秒，需与k单位一致）
    - Evaporated(T)：在时间T内累计蒸发的水分量
    
    该模型适用于食品干燥过程中的水分损失预测，可为工艺优化、干燥时间控制及产品质量管理提供理论支持。
    
    Args:
        initial_moisture: 初始水分含量，单位为质量单位（如 kg 或 g）
        evaporation_rate_constant: 水分蒸发速率常数，单位为 1/时间（如 1/h）
        drying_time: 干燥持续时间，单位需与 evaporation_rate_constant 一致（如 h）

    Returns:
        float: 在干燥时间 T 内累计蒸发的水分量，单位与 initial_moisture 一致
    """
    import math
    # 计算公式：Evaporated(T) = M₀ × (T + (exp(-kT) - 1) / k)
    exponential_term = math.exp(-evaporation_rate_constant * drying_time)
    result = initial_moisture * (drying_time + (exponential_term - 1) / evaporation_rate_constant)
    return result

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run(transport="http", port=8002)
