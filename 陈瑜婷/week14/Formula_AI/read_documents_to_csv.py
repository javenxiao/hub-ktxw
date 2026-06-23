# -*- coding: utf-8 -*-
"""
文档内容读取并保存到CSV文件
功能：读取Documents文件夹下的所有PDF文档内容，并将其保存到CSV文件中
"""

import os
import csv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def extract_text_from_pdf(pdf_path):
    """
    使用PyPDFLoader从PDF文件中提取文本内容
    
    :param pdf_path: PDF文件路径
    :return: 提取的文本内容
    """
    try:
        # 使用PyPDFLoader加载PDF文档
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # 合并所有页面的文本内容
        text = ""
        for page in pages:
            text += page.page_content + "\n"
        
        return text.strip()
    except Exception as e:
        print(f"读取文件 {pdf_path} 时出错: {str(e)}")
        return ""


def update_csv_with_content(csv_file="documents_data.csv", documents_folder="Documents"):
    """
    读取CSV文件，根据filename列从Documents文件夹读取PDF内容并更新content列
    
    :param csv_file: CSV文件路径
    :param documents_folder: 文档所在文件夹路径
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误：CSV文件 {csv_file} 不存在！")
        return
    
    # 检查Documents文件夹是否存在
    if not os.path.exists(documents_folder):
        print(f"错误：文件夹 {documents_folder} 不存在！")
        return
    
    # 读取CSV文件
    data_rows = []
    with open(csv_file, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        # 遍历每一行
        for row in reader:
            filename = row['filename']
            file_path = os.path.join(documents_folder, filename)
            
            # 检查文件是否存在
            if os.path.exists(file_path) and filename.lower().endswith('.pdf'):
                print(f"正在读取: {filename}")
                
                # 提取PDF内容
                content = extract_text_from_pdf(file_path)
                
                # 更新content列
                row['content'] = content
            else:
                print(f"警告：文件 {filename} 不存在或不是PDF文件，跳过")
            
            data_rows.append(row)
    
    # 写回CSV文件
    if data_rows:
        # 使用UTF-8编码写入CSV，避免中文乱码
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据行
            for row in data_rows:
                writer.writerow(row)
        
        print(f"\n成功！共处理 {len(data_rows)} 行数据")
        print(f"内容已更新到: {csv_file}")
    else:
        print("CSV文件为空！")


def read_documents_to_csv(documents_folder="Documents", output_csv="documents_data.csv"):
    """
    读取Documents文件夹下的所有文档内容并保存到CSV文件
    
    :param documents_folder: 文档所在文件夹路径
    :param output_csv: 输出的CSV文件名
    """
    # 确保Documents文件夹存在
    if not os.path.exists(documents_folder):
        print(f"错误：文件夹 {documents_folder} 不存在！")
        return
    
    # 准备数据列表
    data_rows = []
    
    # 遍历Documents文件夹中的所有文件
    for filename in os.listdir(documents_folder):
        file_path = os.path.join(documents_folder, filename)
        
        # 只处理PDF文件
        if filename.lower().endswith('.pdf') and os.path.isfile(file_path):
            print(f"正在读取: {filename}")
            
            # 提取PDF内容
            content = extract_text_from_pdf(file_path)
            
            # 添加到数据列表，formula列暂时为空
            data_rows.append({
                'filename': filename,
                'content': content,
                'formula': ''  # 预留formula列，可以后续填充
            })
    
    # 写入CSV文件
    if data_rows:
        # 使用UTF-8编码写入CSV，避免中文乱码
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['filename', 'content', 'formula']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据行
            for row in data_rows:
                writer.writerow(row)
        
        print(f"\n成功！共处理 {len(data_rows)} 个文档")
        print(f"数据已保存到: {output_csv}")
    else:
        print("未找到任何PDF文档！")


if __name__ == "__main__":
    # 执行CSV内容更新（根据现有CSV文件中的filename读取PDF内容）
    update_csv_with_content()
    
    # 如果需要重新创建CSV文件，取消下面一行的注释
    # read_documents_to_csv()

