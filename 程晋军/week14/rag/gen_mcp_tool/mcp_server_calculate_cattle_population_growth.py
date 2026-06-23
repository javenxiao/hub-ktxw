from fastmcp import FastMCP
import math

# 创建 MCP 服务器实例
mcp = FastMCP("Formula Calculation Server")

@mcp.tool
def calculate_cattle_population_growth(current_population: float, growth_rate: float, carrying_capacity: float) -> float:
    """
    计算牛群数量在逻辑斯蒂增长模型下的下一年数量
    
    公式: N_{t+1} = N_t + r * N_t * (1 - N_t / K)
    
    该模型基于逻辑斯蒂增长思想，描述在环境承载能力限制下的种群动态变化。当牛群数量接近环境承载能力时，增长率逐渐降低，体现资源约束对种群增长的抑制作用。
    
    Args:
        current_population: 当前年份的牛群数量（N_t），单位：头
        growth_rate: 年增长率（r），无量纲，通常取值0~1之间
        carrying_capacity: 环境承载能力（K），即系统可支持的最大牛群数量，单位：头
    
    Returns:
        float: 下一年的预测牛群数量（N_{t+1}），单位：头
    """
    next_population = current_population + growth_rate * current_population * (1 - current_population / carrying_capacity)
    return next_population

if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run(transport="http", port=8002)
