# pip install pymupdf
# pip install -U "pix2tex[api]"
# https://github.com/lukas-blecher/LaTeX-OCR
# pip install sympy
# pip install antlr4-python3-runtime
import asyncio
import os
import time

import fitz
from PIL import Image
from agents import Agent, Runner

from pix2tex.cli import LatexOCR
import re

from sympy.parsing.latex import parse_latex
from sympy import symbols, simplify

from openai import OpenAI


def calculate_formula_score(text):
    """计算文本的「公式得分」（0-100），得分越高越可能是公式"""
    text = text.strip()
    if not text:
        return 0
    total_chars = len(text)
    features = {}
    # 1. 定义特征集合
    # 数学符号（扩展版）
    math_symbols = set("∑∫π√∞∏∇≠≥≤×÷∝∂△Ωθφσμλ∂∆²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉±∓∪∩∈∉⊂⊃≤≥≡≈≅≠∼∽∝∫∬∭∮∇∆∏∑⋅")
    # 基础运算符
    operators = set("+-*/=^%<>()[]{}")
    # 中文字符
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    # 英文单词（排除单个字母，如x/y）
    english_words = re.findall(r"[a-zA-Z]{3,}", text)  # 至少3个字母的英文单词
    # 2. 计算核心特征
    # 数学符号数量
    math_symbol_count = len([c for c in text if c in math_symbols])
    features["math_symbol_ratio"] = math_symbol_count / total_chars if total_chars > 0 else 0
    # 运算符数量
    operator_count = len([c for c in text if c in operators])
    features["operator_density"] = operator_count / total_chars if total_chars > 0 else 0
    # 文本特征占比（中文+长英文单词）
    text_char_count = len(chinese_chars) + sum(len(word) for word in english_words)
    features["text_ratio"] = text_char_count / total_chars if total_chars > 0 else 0
    # 上下标/分式等结构（有则加1，无则0）
    features["has_structure"] = 1 if re.search(r"[²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉]|\\frac|\\sqrt|\\int", text) else 0
    # 3. 计算公式得分（权重可调整）
    score = 0
    score += features["math_symbol_ratio"] * 40  # 数学符号占比：权重40
    score += features["operator_density"] * 40  # 运算符密度：权重40
    score += features["has_structure"] * 20  # 公式结构：权重20
    score -= features["text_ratio"] * 50  # 文本占比：扣分项，权重50
    # 得分限制在0-100之间
    score = max(0, min(100, score))
    return score


def merge_all_intersected_rects(rect_list: list[fitz.Rect]) -> list[fitz.Rect]:
    """合并列表中所有相交/重叠的矩形"""
    # 第一步：过滤空矩形（面积≤0的无效矩形）
    valid_rects = [r for r in rect_list if r.get_area() > 0]
    if len(valid_rects) <= 1:
        return valid_rects
    # 第二步：初始化合并列表，逐个处理矩形
    merged_rects = []
    for rect in valid_rects:
        # 标记是否与现有合并矩形相交
        merged = False
        # 遍历已合并的矩形，检查是否相交
        for i in range(len(merged_rects)):
            existing_rect = merged_rects[i]
            # 判断是否相交
            if existing_rect.intersects(rect) or rect.intersects(existing_rect):
                # 合并为外接矩形
                merged_rects[i] = existing_rect | rect
                merged = True
                # 合并后需重新检查新矩形是否与其他已合并矩形相交（解决链式相交）
                # 例如：合并A+B后，新的A+B可能与C相交，需再次遍历
                j = 0
                while j < len(merged_rects) - 1:
                    current = merged_rects[j]
                    next_rect = merged_rects[j + 1]
                    if current.intersects(next_rect) or next_rect.intersects(current):
                        # 合并相邻的相交矩形
                        merged_rects[j] = current | next_rect
                        del merged_rects[j + 1]
                    else:
                        j += 1
                break
        # 若未与任何已合并矩形相交，加入列表
        if not merged:
            merged_rects.append(rect)
    # 返回
    return merged_rects


def is_strictly_overlap(rect1: fitz.Rect, rect2: fitz.Rect) -> bool:
    """计算是否相交"""
    intersection = rect1 & rect2  # 计算交集矩形
    intersection2 = rect2 & rect1
    return intersection.get_area() > 0 or intersection2.get_area() > 0 # 交集面积>0则为严格重叠


def crop_formula_image(page: fitz.Page, formula_rect: fitz.Rect, save_path: str):
    """裁剪公式区域并保存图片"""
    # 扩展矩形（避免公式边缘被截断，可调整扩展像素）
    expand_pix = 2  # 向外扩展2像素
    crop_rect = formula_rect + (-expand_pix, -expand_pix, expand_pix, expand_pix)
    # 生成图片（可选参数：dpi控制清晰度）
    pix = page.get_pixmap(clip=crop_rect, dpi=300)  # dpi越高越清晰，文件越大
    # 保存图片（支持png/jpg等）
    pix.save(save_path)


def parse_pdf(path: str, id: str) -> list:
    """
    解析 PDF 文件
    :param path: PDF文件路径
    :return: 建模背景、建模公式、公式解析
    """
    model = LatexOCR()
    describe_list = list()
    latex_list = list()
    latex_explanation_list = list()
    with fitz.open(path) as pdf:
        get_describe = False
        get_latex = False
        # 每一页解析
        for page in pdf:
            # 解析
            # 获取所有文本块
            blocks = page.get_text("blocks")
            formula_list = list()
            for block in blocks:
                x0, y0, x1, y1, text, block_no, line_no = block
                # 判断类型
                if "建模背景" in text:
                    get_describe = True
                    get_latex = False
                    continue
                elif "建模公式" in text:
                    get_describe = False
                    get_latex = True
                    continue
                # 获取对应的内容
                if get_describe:
                    # 建模背景
                    describe_list.append(text)
                elif get_latex:
                    # 建模公式 公式解释
                    formula_score = calculate_formula_score(text)
                    if formula_score > 0:
                        # 建模公式
                        formula_list.append(
                            fitz.Rect(x0, y0, x1, y1),  # 公式区域矩形
                        )
                    else:
                        # 公式解释
                        latex_explanation_list.append(text)
            # 建模公式处理
            # ORC 公式
            item_list = list()
            formula_rect_list = merge_all_intersected_rects(formula_list)
            for index, formula_rect in enumerate(formula_rect_list):
                path = f"parse_images/{id}-{index}.png"
                # 裁剪成图片
                crop_formula_image(page, formula_rect, path)
                # 读取图片
                image = Image.open(path)
                latex = model(image)
                item_list.append(latex)
                # 删除图片
                # os.remove(path)
            latex_list += item_list
    # 建模背景处理
    describe = "".join(describe_list)
    # 公式解释处理
    latex_explanation = "".join(latex_explanation_list)
    # 结果
    result = {
        "describe": describe,
        "latex": latex_list,
        "latex_explanation": latex_explanation,
    }
    return result


client = OpenAI(
    api_key="sk-b15ba6a031b447f7b18d195b834629f1",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


async def create_mcp_tool(result, index):
    describe = result["describe"]
    latex = result["latex"][0]
    latex_explanation = result["latex_explanation"]
    print("describe", describe)
    print("latex", latex)
    print("latex_explanation", latex_explanation)
    # 生成MCP Tool助手
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": f"""
你是一个资深的程序员，可以根据建模背景、建模公式和公式解释，使用python语言写出mcp tool的相关代码。
要求使用sympy库进行公式计算，函数有多少个参数，就是用symbols定义多少个符号变量，参数名称和符号变量不要重复，避免冲突，一步一步地拼接，直到构建完整的公式，然后把参数带入公式计算得到结果。
输出的代码为markdown格式，代码必须可正常运行。只要输出代码就行，不要任何说明和举例。
输出格式：
```
@mcp.tool
def 函数名称(
    参数1: Annotated[参数1类型, 参数1说明],
    ...
):
    \"\"\"函数说明\"\"\"
    函数实现
```
"""},
            {"role": "user", "content": f"""
建模背景: {describe}
建模公式: {latex}
公式解释: {latex_explanation}
"""}
        ]
    )
    data = completion.choices[0].message.content
    print(data)
    with open(f"parse_results/{index}.txt", "w", encoding="utf-8") as f:
        f.write("建模背景：\n")
        f.write(describe)
        f.write("\n")
        f.write("建模公式：\n")
        f.write(latex)
        f.write("\n")
        f.write("公式解释：\n")
        f.write(latex_explanation)
        f.write("\n")
        f.write("MCP Tool 代码：\n")
        f.write(data)
        f.write("\n")

pdfs = [
    "documents/0ba15b17-85d2-4944-9a04-a9bd23c2e3f.pdf",
    "documents/0a948fc4-b083-44c6-af02-70be51108f7.pdf",
    "documents/00ac792a-04dd-4639-abbd-d7f78cbb7ea.pdf",
    "documents/2aa97691-9154-44b2-ae23-ce32410b273.pdf",
    "documents/0daef473-e660-4984-be4d-940433aa889.pdf",
    "documents/3a9ec7e4-3d97-4c10-ab97-61f27e0ec28.pdf",
    "documents/4b5be424-d194-4017-9fe8-17a3425e042.pdf",
    "documents/4a595f0e-ee88-4f68-96e4-47e90e2b7cb.pdf",
    "documents/4d27723a-1f09-4ae5-b9be-4eeded91bd3.pdf",
    "documents/2e07a9dc-b19e-4034-a72d-d7280e5db0b.pdf",
]

if __name__ == '__main__':
    for index, path in enumerate(pdfs):
        result = parse_pdf(path, str(index))
        asyncio.run(create_mcp_tool(result, str(index)))

