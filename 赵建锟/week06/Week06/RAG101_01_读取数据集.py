import json  # 导入json模块，用于处理JSON数据

# pip install --upgrade charset-normalizer
# pip install pdfplumber
import pdfplumber  # 导入pdfplumber模块，用于处理PDF文件

# 加载存储问题的JSON文件
with open("questions.json", "r",encoding='UTF-8') as f:
    questions = json.load(f)
print("questions.json")  # 打印提示信息，显示加载的JSON文件名
print(questions[0])  # 打印第一个问题的内容

print("\n")  # 打印空行

# 打开PDF文件
pdf = pdfplumber.open("汽车知识手册.pdf")
print("pages: ", len(pdf.pages))  # 打印提示信息，显示PDF文件的页数
print(pdf.pages[0].extract_text())  # 提取并打印第一页的文本内容
