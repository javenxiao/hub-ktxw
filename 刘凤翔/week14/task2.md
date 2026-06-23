# PDF公式解析与MCP工具定义

## 一、PDF文档解析结果

### 文档1：数学分析教材
**提取公式1**：$\lim_{x \to a} \frac{f(x) - f(a)}{x - a} = f'(a)$
- **公式类型**：导数定义
- **应用场景**：计算函数在某点的导数

### 文档2：物理力学教材
**提取公式2**：$F = ma$
- **公式类型**：牛顿第二定律
- **应用场景**：计算物体受力

### 文档3：统计学教材
**提取公式3**：$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$
- **公式类型**：均值公式
- **应用场景**：计算平均值

### 文档4：金融数学文档
**提取公式4**：$A = P(1 + r)^t$
- **公式类型**：复利计算公式
- **应用场景**：计算投资未来价值

### 文档5：工程材料学
**提取公式5**：$\sigma = \frac{F}{A}$
- **公式类型**：应力公式
- **应用场景**：计算材料应力

### 文档6：热力学教材
**提取公式6**：$Q = mc\Delta T$
- **公式类型**：热量计算公式
- **应用场景**：计算热量变化

### 文档7：电磁学教材
**提取公式7**：$V = IR$
- **公式类型**：欧姆定律
- **应用场景**：计算电压

### 文档8：几何学教材
**提取公式8**：$A = \pi r^2$
- **公式类型**：圆面积公式
- **应用场景**：计算圆面积

### 文档9：概率论教材
**提取公式9**：$P(A|B) = \frac{P(A \cap B)}{P(B)}$
- **公式类型**：条件概率公式
- **应用场景**：计算条件概率

### 文档10：量子力学教材
**提取公式10**：$E = h\nu$
- **公式类型**：光子能量公式
- **应用场景**：计算光子能量

## 二、MCP工具定义

### 工具1：导数计算工具
```python
def calculate_derivative(formula: str, f_expression: str, a: float, x_val: float = None) -> dict:
    """
    计算函数在某点的导数
    
    参数:
        formula: 导数定义公式
        f_expression: 函数表达式，如 "x**2 + 3*x + 5"
        a: 求导点
        x_val: 计算f(x)的x值（可选）
    
    返回:
        dict: 包含导数值和计算过程
    """
    try:
        import sympy as sp
        
        # 解析函数表达式
        x = sp.symbols('x')
        f = sp.sympify(f_expression)
        
        # 计算导数（使用公式定义）
        f_prime = sp.diff(f, x)
        
        # 在点a处求值
        derivative_at_a = float(f_prime.subs(x, a))
        
        # 如果提供了x_val，计算f(x)
        f_x_val = None
        if x_val is not None:
            f_x_val = float(f.subs(x, x_val))
        
        result = {
            "formula": formula,
            "function": str(f),
            "derivative": str(f_prime),
            "derivative_at_a": derivative_at_a,
            "point": a
        }
        
        if f_x_val is not None:
            result["f_at_x"] = f_x_val
            
        return result
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具2：牛顿第二定律计算工具
```python
def calculate_force(formula: str, m: float, a: float) -> dict:
    """
    根据牛顿第二定律计算力
    
    参数:
        formula: F = ma
        m: 质量(kg)
        a: 加速度(m/s²)
    
    返回:
        dict: 包含力和单位信息
    """
    try:
        force = m * a
        
        return {
            "formula": formula,
            "mass": m,
            "acceleration": a,
            "force": force,
            "units": {
                "force": "N (Newtons)",
                "mass": "kg",
                "acceleration": "m/s²"
            },
            "explanation": f"质量为{m}kg的物体以{a}m/s²的加速度运动时，受到的力为{force}N"
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具3：均值计算工具
```python
def calculate_mean(formula: str, data: list) -> dict:
    """
    计算数据集的平均值
    
    参数:
        formula: 均值公式
        data: 数值列表
    
    返回:
        dict: 包含均值、总和和数据信息
    """
    try:
        import numpy as np
        
        data_array = np.array(data)
        n = len(data_array)
        total = np.sum(data_array)
        mean_value = total / n
        
        return {
            "formula": formula,
            "data": data,
            "n": n,
            "sum": float(total),
            "mean": float(mean_value),
            "data_info": {
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "std": float(np.std(data_array))
            }
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具4：复利计算工具
```python
def calculate_compound_interest(formula: str, P: float, r: float, t: float) -> dict:
    """
    计算复利
    
    参数:
        formula: A = P(1 + r)^t
        P: 本金
        r: 年利率（小数形式）
        t: 时间（年）
    
    返回:
        dict: 包含未来价值和详细计算
    """
    try:
        import numpy as np
        
        # 计算未来价值
        A = P * (1 + r) ** t
        
        # 计算利息
        interest = A - P
        
        # 生成逐年增长数据
        years = list(range(int(t) + 1))
        values = [P * (1 + r) ** year for year in years]
        
        return {
            "formula": formula,
            "principal": P,
            "rate": r,
            "time": t,
            "future_value": A,
            "interest_earned": interest,
            "growth_data": {
                "years": years,
                "values": values
            },
            "explanation": f"本金{P}元，年利率{r*100}%，经过{t}年后增长为{A:.2f}元"
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具5：应力计算工具
```python
def calculate_stress(formula: str, F: float, A: float) -> dict:
    """
    计算材料应力
    
    参数:
        formula: σ = F/A
        F: 力(N)
        A: 面积(m²)
    
    返回:
        dict: 包含应力和安全系数
    """
    try:
        stress = F / A
        
        # 常用材料的屈服强度（Pa）
        material_strength = {
            "steel": 250e6,
            "aluminum": 70e6,
            "concrete": 30e6,
            "wood": 10e6
        }
        
        safety_factors = {}
        for material, strength in material_strength.items():
            if stress > 0:
                safety_factors[material] = strength / stress
        
        return {
            "formula": formula,
            "force": F,
            "area": A,
            "stress": stress,
            "units": {
                "stress": "Pa (Pascals)",
                "force": "N",
                "area": "m²"
            },
            "safety_factors": safety_factors,
            "interpretation": f"应力为{stress:.2e}Pa，每平方米承受{F/A:.2f}牛顿的力"
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具6：热量计算工具
```python
def calculate_heat(formula: str, m: float, c: float, delta_T: float) -> dict:
    """
    计算热量变化
    
    参数:
        formula: Q = mcΔT
        m: 质量(kg)
        c: 比热容(J/(kg·K))
        delta_T: 温度变化(K)
    
    返回:
        dict: 包含热量和能量转换
    """
    try:
        Q = m * c * delta_T
        
        # 转换为其他能量单位
        energy_conversion = {
            "joules": Q,
            "kilojoules": Q / 1000,
            "calories": Q / 4.184,
            "kilocalories": Q / 4184,
            "kWh": Q / 3.6e6
        }
        
        return {
            "formula": formula,
            "mass": m,
            "specific_heat": c,
            "temperature_change": delta_T,
            "heat": Q,
            "energy_units": energy_conversion,
            "examples": {
                "boil_water": f"将{m}kg水加热{delta_T}K需要{Q:.2f}J能量",
                "equivalent_mass": f"相当于{(Q/9e16)*1000:.2e}克物质的质能"
            }
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具7：欧姆定律计算工具
```python
def calculate_ohms_law(formula: str, **kwargs) -> dict:
    """
    根据欧姆定律计算电压、电流或电阻
    
    参数:
        formula: V = IR
        kwargs: 提供任意两个参数(V, I, R)
    
    返回:
        dict: 包含所有三个参数的值
    """
    try:
        # 检查输入参数
        provided = set(kwargs.keys())
        required = {'V', 'I', 'R'}
        
        if len(provided) != 2:
            return {"error": "需要提供任意两个参数(V, I, R)"}
        
        # 计算缺失的参数
        if 'V' not in provided and 'I' in provided and 'R' in provided:
            V = kwargs['I'] * kwargs['R']
            I = kwargs['I']
            R = kwargs['R']
        elif 'I' not in provided and 'V' in provided and 'R' in provided:
            V = kwargs['V']
            I = kwargs['V'] / kwargs['R']
            R = kwargs['R']
        elif 'R' not in provided and 'V' in provided and 'I' in provided:
            V = kwargs['V']
            I = kwargs['I']
            R = kwargs['V'] / kwargs['I']
        else:
            return {"error": "无效的参数组合"}
        
        # 计算功率
        P = V * I
        
        return {
            "formula": formula,
            "voltage": V,
            "current": I,
            "resistance": R,
            "power": P,
            "units": {
                "voltage": "V (Volts)",
                "current": "A (Amperes)",
                "resistance": "Ω (Ohms)",
                "power": "W (Watts)"
            },
            "circuit_info": {
                "type": "串联" if R > 0 else "未知",
                "power_consumption": f"{P}W"
            }
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具8：几何面积计算工具
```python
def calculate_circle_area(formula: str, r: float) -> dict:
    """
    计算圆面积
    
    参数:
        formula: A = πr²
        r: 半径
    
    返回:
        dict: 包含面积、周长和相关几何信息
    """
    try:
        import numpy as np
        import sympy as sp
        
        # 使用numpy计算数值
        A_numpy = np.pi * r ** 2
        circumference = 2 * np.pi * r
        
        # 使用sympy保持符号精度
        r_sym = sp.symbols('r')
        A_sympy = sp.pi * r_sym ** 2
        
        return {
            "formula": formula,
            "radius": r,
            "area_numpy": float(A_numpy),
            "area_symbolic": str(A_sympy.subs(r_sym, r)),
            "circumference": float(circumference),
            "diameter": 2 * r,
            "geometric_ratios": {
                "area_to_circumference": A_numpy / circumference,
                "circumference_to_diameter": circumference / (2 * r)
            },
            "comparisons": {
                "square_equivalent_side": np.sqrt(A_numpy),
                "inscribed_square_area": 2 * r ** 2
            }
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具9：概率计算工具
```python
def calculate_conditional_probability(formula: str, P_A: float, P_B_given_A: float, P_B: float = None) -> dict:
    """
    计算条件概率
    
    参数:
        formula: P(A|B) = P(A∩B) / P(B)
        P_A: 事件A的概率
        P_B_given_A: 在A发生的条件下B发生的概率
        P_B: 事件B的概率（可选，如果不提供则计算）
    
    返回:
        dict: 包含各种概率值
    """
    try:
        # 计算联合概率 P(A∩B)
        P_A_and_B = P_A * P_B_given_A
        
        # 如果未提供P_B，假设独立性计算
        if P_B is None:
            P_B = P_A_and_B / P_A if P_A > 0 else 0
        
        # 计算条件概率 P(A|B)
        if P_B > 0:
            P_A_given_B = P_A_and_B / P_B
        else:
            P_A_given_B = 0
        
        # 计算其他相关概率
        independence_check = abs(P_A_and_B - P_A * P_B) < 1e-10
        
        return {
            "formula": formula,
            "probabilities": {
                "P(A)": P_A,
                "P(B|A)": P_B_given_A,
                "P(B)": P_B,
                "P(A∩B)": P_A_and_B,
                "P(A|B)": P_A_given_B
            },
            "statistics": {
                "independence": independence_check,
                "correlation": (P_A_and_B - P_A * P_B) / np.sqrt(P_A * (1-P_A) * P_B * (1-P_B)) if P_A*(1-P_A)*P_B*(1-P_B) > 0 else 0
            },
            "interpretation": {
                "prior_odds": P_A / (1 - P_A) if P_A < 1 else float('inf'),
                "posterior_odds": P_A_given_B / (1 - P_A_given_B) if P_A_given_B < 1 else float('inf'),
                "bayes_factor": P_A_given_B / P_A if P_A > 0 else float('inf')
            }
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

### 工具10：光子能量计算工具
```python
def calculate_photon_energy(formula: str, nu: float = None, lam: float = None) -> dict:
    """
    计算光子能量
    
    参数:
        formula: E = hν
        nu: 频率(Hz) - 提供nu或lam之一
        lam: 波长(m) - 提供nu或lam之一
    
    返回:
        dict: 包含光子能量和相关物理量
    """
    try:
        import numpy as np
        
        # 物理常数
        h = 6.62607015e-34  # 普朗克常数 (J·s)
        c = 299792458      # 光速 (m/s)
        
        # 计算频率或波长
        if nu is not None:
            frequency = nu
            wavelength = c / nu if nu > 0 else float('inf')
        elif lam is not None:
            wavelength = lam
            frequency = c / lam if lam > 0 else 0
        else:
            return {"error": "需要提供频率(nu)或波长(lam)"}
        
        # 计算能量
        E = h * frequency
        
        # 转换为电子伏特
        E_eV = E / 1.602176634e-19
        
        # 根据能量分类
        radiation_type = "未知"
        if E_eV < 0.01:
            radiation_type = "无线电波"
        elif E_eV < 0.1:
            radiation_type = "微波"
        elif E_eV < 1:
            radiation_type = "红外线"
        elif E_eV < 3:
            radiation_type = "可见光"
        elif E_eV < 100:
            radiation_type = "紫外线"
        elif E_eV < 100000:
            radiation_type = "X射线"
        else:
            radiation_type = "伽马射线"
        
        return {
            "formula": formula,
            "energy_joules": E,
            "energy_eV": E_eV,
            "frequency": frequency,
            "wavelength": wavelength,
            "physical_constants": {
                "planck_constant": h,
                "speed_of_light": c
            },
            "radiation_type": radiation_type,
            "color_info": self._get_color_from_wavelength(wavelength) if lam is not None else None
        }
        
    except Exception as e:
        return {"error": str(e), "formula": formula}
```

## 三、MCP工具使用示例

### 示例1：使用导数计算工具
```python
# 计算函数 f(x) = x^2 + 3x + 5 在 x=2 处的导数
result = calculate_derivative(
    formula="lim_{x→a} (f(x)-f(a))/(x-a) = f'(a)",
    f_expression="x**2 + 3*x + 5",
    a=2
)
print(f"导数: {result['derivative']}")
print(f"在x=2处的导数值: {result['derivative_at_a']}")
```

### 示例2：使用复利计算工具
```python
# 计算1000元本金，5%年利率，10年后的价值
result = calculate_compound_interest(
    formula="A = P(1+r)^t",
    P=1000,
    r=0.05,
    t=10
)
print(f"未来价值: {result['future_value']:.2f}元")
print(f"赚取利息: {result['interest_earned']:.2f}元")
```

### 示例3：使用欧姆定律工具
```python
# 已知电压12V，电阻4Ω，求电流
result = calculate_ohms_law(
    formula="V = IR",
    V=12,
    R=4
)
print(f"电流: {result['current']}A")
print(f"功率: {result['power']}W")
```

## 四、工具集成与扩展

### 4.1 工具注册表
```python
class FormulaToolsRegistry:
    def __init__(self):
        self.tools = {
            "derivative_calculator": calculate_derivative,
            "newtons_law": calculate_force,
            "mean_calculator": calculate_mean,
            "compound_interest": calculate_compound_interest,
            "stress_calculator": calculate_stress,
            "heat_calculator": calculate_heat,
            "ohms_law": calculate_ohms_law,
            "circle_area": calculate_circle_area,
            "conditional_probability": calculate_conditional_probability,
            "photon_energy": calculate_photon_energy
        }
    
    def execute_tool(self, tool_name: str, **kwargs):
        """执行指定工具"""
        if tool_name in self.tools:
            try:
                return self.tools[tool_name](**kwargs)
            except Exception as e:
                return {"error": f"工具执行失败: {str(e)}"}
        else:
            return {"error": f"未知工具: {tool_name}"}
```

### 4.2 API接口示例
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
registry = FormulaToolsRegistry()

class ToolRequest(BaseModel):
    tool_name: str
    parameters: dict

@app.post("/execute_tool")
async def execute_tool(request: ToolRequest):
    """执行公式计算工具的API接口"""
    result = registry.execute_tool(request.tool_name, **request.parameters)
    return {
        "success": "error" not in result,
        "result": result
    }
```

## 五、总结

本文定义了10个基于PDF文档解析公式的MCP工具，每个工具都具备：

1. **明确的输入参数**：根据公式定义必需的参数
2. **完整的计算逻辑**：使用numpy和sympy进行计算
3. **丰富的返回信息**：包含计算结果、单位、解释和附加信息
4. **错误处理机制**：确保工具的健壮性
5. **实际应用场景**：每个工具都解决特定的数学或物理问题

这些工具可以集成到更大的系统中，为学术研究、工程计算、金融分析等领域提供专业的公式计算能力。通过统一的API接口，用户可以方便地调用这些工具进行复杂的数学计算。