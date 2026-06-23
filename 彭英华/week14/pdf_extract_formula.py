import fitz
from openai import OpenAI
import json
import os
#读取 PDF
num=0
path_list=[]
data_list=[]
# 选取10个pdf
for dir in os.listdir('./documents'):
    if num==10:
        break
    if dir.lower().endswith(".md"):
        continue
    path = "./documents/" + dir
    path_list.append(path)
    num+=1
print(path_list)
用PyMuPDF提取pdf的文本内容
for path in path_list:
    page = fitz.open(path)
    full_content = ""
    for text in page:
        full_content+=text.get_text()
        full_content+="\n\n"
    # 用大模型从文档中提取出公式，公式描述和公式所需参数
    model = OpenAI(api_key="sk-90aa2b7df82745f3a46373cc0ddd0497",
                           base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    template = """
            你是一个公式和信息抽取专家，现在你需要根据用户提供的关于公式的文档内容，从中抽取出公式描述，核心数学公式，公式所需的参数以及参数的说明，并填充下面的json，你的回答必须是且仅是一个**格式完全正确的JSON对象**
            ```json
                {{
                  “formula_description":"在这里对公式进行描述，说明公式是做什么用的，具体有什么功能"
                  "structured_data": {{
                    "formula": "这里填入你从文档中找到的核心数学公式。",
                    "parameters": {{ "变量名1": "变量含义" }},
                  }}
                }}
                ```
            """
    messages= [{"role":"system","content":template},
                       {"role":"user","content":full_content}]
    result = model.chat.completions.create(model="qwen3-max",messages=messages)
    response = result.choices[0].message.content
    # 解析大模型返回的结果，并存为json文件
    try:
        json_data = response.split('json')[1].strip().split("```")[0]
        data=json.loads(json_data)
        data_list.append(data)
    except Exception as e:
        print("解析json格式错误",e)
with open('formulas_dict.json', 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)
print(f"已保存 {len(data_list)} 个公式到 formulas_dict.json")
