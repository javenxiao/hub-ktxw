# 文档公式解析与智能问答系统实现方案

## 一、整体思路理解

文档公式解析与智能问答的核心思路是：**从文档中提取公式结构化信息，建立检索索引，根据用户问题检索相关公式，最后利用大语言模型进行推理或代码执行得到答案**。

## 二、待选方案分析

### 方案1：OCR-based公式识别
**适用场景**：扫描版PDF、图像公式
- 使用PaddleOCR/MMOCR检测公式区域
- LaTeX-OCR进行公式识别
- 优点：通用性强
- 缺点：准确率受图像质量影响

### 方案2：PDF原生文本提取
**适用场景**：Latex生成的PDF、包含MathML的文档
- 利用PDF中的公式标记（如MathML）
- 使用pdfminer等工具提取原生LaTeX
- 优点：精度高，保持原始格式
- 缺点：依赖PDF内部标记

### 方案3：深度学习端到端模型
**适用场景**：大规模专业文档处理
- 使用预训练模型（如Donut）直接输出LaTeX
- 端到端训练，一体化解决
- 优点：准确性高
- 缺点：需要大量标注数据

## 三、实施步骤详解

### 步骤1：PDF公式解析

**推荐方案：混合解析策略**

```python
class HybridFormulaParser:
    def parse_pdf_formulas(self, pdf_path: str) -> List[Dict]:
        formulas = []
        
        # 1. 优先提取原生LaTeX（方案2）
        native_formulas = self._extract_native_latex(pdf_path)
        
        # 2. 对剩余区域使用OCR识别（方案1）
        formula_regions = self._detect_formula_regions(pdf_path)
        for region in formula_regions:
            if not self._is_duplicate(region, formulas):
                latex = self._ocr_formula(region)
                if latex:
                    formulas.append({
                        'latex': latex,
                        'bbox': region,
                        'structural_info': self._parse_structure(latex),
                        'symbols': self._extract_symbols(latex)
                    })
        
        return formulas

    def _parse_structure(self, latex: str) -> Dict:
        """解析公式结构"""
        return {
            'operators': self._extract_operators(latex),
            'variables': self._extract_variables(latex),
            'functions': self._extract_functions(latex),
            'complexity': self._calculate_complexity(latex)
        }
```

### 步骤2：RAG检索和排序

**推荐方案：三级检索架构**

```python
class MultiStageRetriever:
    """BM25 → 稠密检索 → 交叉编码器重排"""
    
    def __init__(self):
        # 第一级：BM25（快速召回）
        self.bm25 = BM25Okapi()
        
        # 第二级：稠密检索（精确匹配）
        self.dense_model = SentenceTransformer('BAAI/bge-large-zh')
        
        # 第三级：重排序（精细调整）
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def retrieve(self, query: str, formulas: List[Dict], top_k: int = 8) -> List[int]:
        """
        三级检索流程：
        1. BM25召回100个候选
        2. 稠密检索筛选30个
        3. 交叉编码器重排得到top_k
        """
        # 步骤1: BM25粗筛
        bm25_scores = self.bm25.get_scores(query)
        bm25_candidates = np.argsort(bm25_scores)[-100:][::-1]
        
        # 步骤2: 稠密检索
        query_embedding = self.dense_model.encode([query])
        formula_texts = [self._formula_to_text(formulas[i]) for i in bm25_candidates]
        formula_embeddings = self.dense_model.encode(formula_texts)
        
        dense_scores = cosine_similarity(query_embedding, formula_embeddings)[0]
        dense_candidates = bm25_candidates[np.argsort(dense_scores)[-30:][::-1]]
        
        # 步骤3: 交叉编码器重排
        pairs = [(query, formulas[i]['latex']) for i in dense_candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # 最终排序
        final_indices = dense_candidates[np.argsort(rerank_scores)[-top_k:][::-1]]
        return final_indices
```

### 步骤3：Qwen-3 Thinking推理

#### 方案3.1：代码生成执行模式

```python
class CodeGenerationSolver:
    """生成可执行代码进行求解"""
    
    def solve_with_code(self, question: str, formula_latex: str) -> Dict:
        prompt = f'''
你是一个数学问题解决专家。请基于以下公式解决用户的问题。

公式: {formula_latex}

问题: {question}

请按照以下步骤进行：
1. 分析问题与公式的关系
2. 将问题中的条件转换为公式中的变量
3. 生成SymPy/Python代码来求解
4. 确保代码可以直接执行

请生成可执行的Python代码（包含必要的import和print语句）：
'''
        
        # 调用Qwen-3模型
        response = self.qwen_model.generate(prompt)
        
        # 提取并执行代码
        code_blocks = self._extract_code_blocks(response)
        results = []
        
        for code in code_blocks:
            try:
                # 在安全沙箱中执行
                result = self._execute_in_sandbox(code)
                results.append({
                    'code': code,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'code': code,
                    'error': str(e),
                    'success': False
                })
        
        return {
            'method': 'code_generation',
            'formula': formula_latex,
            'results': results,
            'explanation': response
        }
```

#### 方案3.2：直接推理模式

```python
class DirectReasoningSolver:
    """直接推理得到答案"""
    
    def solve_with_reasoning(self, question: str, formula_latex: str) -> Dict:
        prompt = f'''
你是一个数学推理专家。请基于以下公式直接推理出问题的答案。

公式: {formula_latex}

问题: {question}

请按照以下格式输出：
1. 分析公式的含义和应用场景
2. 将问题转化为公式可以处理的形式
3. 逐步推理计算过程
4. 给出最终答案

注意：不要生成代码，只需要逻辑推理。
'''
        
        response = self.qwen_model.generate(prompt)
        
        return {
            'method': 'direct_reasoning',
            'formula': formula_latex,
            'answer': self._extract_answer(response),
            'reasoning_steps': self._extract_reasoning_steps(response),
            'full_response': response
        }
```

#### 推荐方案：自适应混合求解

```python
class AdaptiveFormulaSolver:
    """根据问题类型选择最佳求解策略"""
    
    def solve(self, question: str, formulas: List[Dict]) -> Dict:
        # 1. 判断问题类型
        problem_type = self._classify_problem(question)
        
        # 2. 尝试不同求解策略
        solutions = []
        
        # 优先尝试代码生成（适合数值计算）
        if problem_type in ['numerical', 'calculation', 'evaluation']:
            for formula in formulas[:3]:
                solution = self._try_code_solution(question, formula['latex'])
                if solution['success']:
                    solutions.append(solution)
                    break
        
        # 代码失败则尝试符号计算
        if not solutions and problem_type in ['algebraic', 'symbolic']:
            for formula in formulas[:2]:
                solution = self._try_symbolic_solution(question, formula['latex'])
                if solution['success']:
                    solutions.append(solution)
                    break
        
        # 最后尝试直接推理
        if not solutions:
            for formula in formulas[:1]:
                solution = self._try_reasoning_solution(question, formula['latex'])
                solutions.append(solution)
        
        # 3. 选择最佳解决方案
        best_solution = self._select_best_solution(solutions)
        return best_solution
```

## 四、完整系统集成

```python
class FormulaQASystem:
    """完整的文档公式问答系统"""
    
    def __init__(self, pdf_path: str):
        # 初始化所有模块
        self.parser = HybridFormulaParser()
        self.retriever = MultiStageRetriever()
        self.solver = AdaptiveFormulaSolver()
        
        # 处理PDF文档
        print("正在解析PDF文档...")
        self.formulas = self.parser.parse_pdf_formulas(pdf_path)
        print(f"成功提取 {len(self.formulas)} 个公式")
        
        # 构建检索索引
        print("正在构建检索索引...")
        self.retriever.index_formulas(self.formulas)
    
    def ask(self, question: str) -> Dict:
        """主问答接口"""
        print(f"\n用户问题: {question}")
        
        # 步骤1: 检索相关公式
        print("正在检索相关公式...")
        formula_indices = self.retriever.retrieve(question, self.formulas, top_k=5)
        relevant_formulas = [self.formulas[i] for i in formula_indices]
        
        print(f"找到 {len(relevant_formulas)} 个相关公式:")
        for i, formula in enumerate(relevant_formulas, 1):
            print(f"  {i}. {formula['latex'][:100]}...")
        
        # 步骤2: 求解
        print("\n正在求解...")
        solution = self.solver.solve(question, relevant_formulas)
        
        # 返回结果
        return {
            'question': question,
            'relevant_formulas': relevant_formulas[:3],  # 只返回前3个最相关的
            'solution': solution,
            'confidence': self._calculate_confidence(solution)
        }
    
    def _calculate_confidence(self, solution: Dict) -> float:
        """计算答案置信度"""
        method_scores = {
            'code_generation': 0.9,
            'symbolic_calculation': 0.8,
            'direct_reasoning': 0.7
        }
        
        base_score = method_scores.get(solution['method'], 0.5)
        
        # 根据代码执行是否成功调整分数
        if solution['method'] == 'code_generation' and solution.get('execution_success'):
            return min(base_score + 0.1, 1.0)
        
        return base_score
```

## 五、使用示例

```python
# 初始化系统
system = FormulaQASystem("math_textbook.pdf")

# 提问
question = "计算当x=2时，函数f(x)=x²+3x+5的值是多少？"
result = system.ask(question)

# 输出结果
print(f"\n=== 答案 ===")
print(f"问题: {result['question']}")
print(f"使用的公式: {result['solution']['formula']}")
print(f"求解方法: {result['solution']['method']}")
print(f"答案: {result['solution']['answer']}")
print(f"置信度: {result['confidence']:.2f}")

# 显示详细推理过程
if result['solution'].get('reasoning_steps'):
    print("\n推理步骤:")
    for step in result['solution']['reasoning_steps']:
        print(f"  - {step}")
```

## 六、优化建议

1. **性能优化**
   - 缓存公式嵌入向量
   - 使用FAISS进行快速相似度搜索
   - 并行处理多个公式解析

2. **准确性提升**
   - 加入公式验证步骤
   - 使用多模型投票机制
   - 建立公式知识图谱

3. **安全性增强**
   - 代码执行沙箱隔离
   - 限制资源使用（CPU/内存）
   - 输入输出过滤

## 七、总结

本方案实现了一个完整的文档公式解析与智能问答系统，具有以下特点：

1. **灵活的公式解析**：混合使用多种技术，适应不同类型的PDF文档
2. **精准的公式检索**：三级检索架构确保找到最相关的公式
3. **智能的求解策略**：根据问题类型自适应选择最佳求解方法
4. **安全的代码执行**：在沙箱环境中执行生成的代码
5. **可解释的推理过程**：提供详细的推理步骤和置信度评估

该系统可广泛应用于教育、科研、技术文档等领域，帮助用户快速理解和应用文档中的数学公式。