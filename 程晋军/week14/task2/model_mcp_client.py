import asyncio
from fastmcp import Client

# 创建客户端实例，连接到MCP服务器
client = Client("http://localhost:8002/mcp")

async def demo_all_models():
    """演示调用所有模型函数"""
    async with client:
        print("=== 模型函数调用演示 ===\n")
        
        # 1. 溶解氧浓度计算
        print("1. 溶解氧浓度计算:")
        result = await client.call_tool("calculate_dissolved_oxygen", {
            "t": 1.0,
            "a": 5.0,
            "b": 0.1,
            "c": 2.0,
            "d": 0.5
        })
        print(f"   结果: {result}\n")
        
        # 2. 电商订单预测
        print("2. 电商订单预测:")
        result = await client.call_tool("predict_orders", {
            "ad_spend": 1000.0,
            "discount_rate": 0.1,
            "prev_orders": 50.0
        })
        print(f"   结果: {result}\n")
        
        # 3. 食品干燥过程中水分含量计算
        print("3. 食品干燥过程中水分含量计算:")
        result = await client.call_tool("calculate_moisture_content", {
            "t": 2.0,
            "M0": 10.0,
            "k": 0.5
        })
        print(f"   结果: {result}\n")
        
        # 4. 食品干燥过程中累计水分蒸发量计算
        print("4. 食品干燥过程中累计水分蒸发量计算:")
        result = await client.call_tool("calculate_evaporated_amount", {
            "T": 5.0,
            "M0": 10.0,
            "k": 0.3
        })
        print(f"   结果: {result}\n")
        
        # 5. 农作物产量估算
        print("5. 农作物产量估算:")
        result = await client.call_tool("estimate_crop_yield", {
            "F": 8.0,
            "I": 20.0,
            "T": 25.0,
            "a": 10.0,
            "b": 5.0,
            "c": 0.1
        })
        print(f"   结果: {result}\n")
        
        # 6. 学生学习效果评估
        print("6. 学生学习效果评估:")
        result = await client.call_tool("evaluate_student_score", {
            "x1": 10.0,  # 学习时长
            "x2": 90.0,  # 出勤率
            "x3": 85.0,  # 平时测验平均分
            "x4": 4.0,   # 课堂参与度
            "w1": 0.3,   # 学习时长权重
            "w2": 0.2,   # 出勤率权重
            "w3": 0.4,   # 测验分数权重
            "w4": 0.1    # 参与度权重
        })
        print(f"   结果: {result}\n")
        
        # 7. 确定性模型（二次函数）
        print("7. 确定性模型（二次函数）:")
        result = await client.call_tool("compute_deterministic_model", {
            "x": 3.0
        })
        print(f"   结果: {result}\n")
        
        # 8. 传播影响力模型
        print("8. 传播影响力模型:")
        result = await client.call_tool("calculate_influence", {
            "content_quality": 8.0,
            "channels": 5.0,
            "engagement": 7.0,
            "time": 10.0
        })
        print(f"   结果: {result}\n")
        
        # 9. 牛群数量演化模型
        print("9. 牛群数量演化模型:")
        result = await client.call_tool("predict_cattle_population", {
            "N_t": 100.0,
            "r": 0.1,
            "K": 500.0
        })
        print(f"   结果: {result}\n")


async def call_specific_tool(tool_name: str, params: dict):
    """调用指定的工具函数"""
    async with client:
        try:
            result = await client.call_tool(tool_name, params)
            print(f"调用 {tool_name} 的结果: {result}")
            return result
        except Exception as e:
            print(f"调用 {tool_name} 时出错: {e}")
            return None


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_all_models())
    
    # 示例：单独调用某个工具
    # asyncio.run(call_specific_tool("calculate_dissolved_oxygen", {
    #     "t": 2.0,
    #     "a": 5.0,
    #     "b": 0.1,
    #     "c": 2.0,
    #     "d": 0.5
    # }))