import requests
import os
import time
import json
import zipfile
import shutil
from pathlib import Path

# 配置参数
token = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI0MjcwMDk3NyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2NDU5ODY0NiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTY2MDYxNzU2NzAiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJmNWM0NWY2ZS1kNDBjLTQ4YjEtYTZmOS04NmM1ZTM2MjgyODEiLCJlbWFpbCI6IiIsImV4cCI6MTc2NTgwODI0Nn0.aF9sxT1_tyOzGVr6eq9vI_DfC-Y02hm4hofYDmHyT89hFH4g2tUIoAozfdPEYzX_Yh1sMStUZsjVnot9WwqtNw"
upload_url_api = "https://mineru.net/api/v4/file-urls/batch"
model_version = "vlm"

def batch_upload_pdfs(folder_path):
    """
    批量上传PDF文件并获取URL
    
    Args:
        folder_path (str): 包含PDF文件的文件夹路径
        
    Returns:
        dict: 包含batch_id和每个文件信息的字典
    """
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    try:
        # 1. 遍历文件夹，获取所有PDF文件
        pdf_files = []
        for filename in os.listdir(folder_path):
            file_full_path = os.path.join(folder_path, filename)
            # 只保留PDF文件且是文件（不是文件夹）
            if os.path.isfile(file_full_path) and filename.lower().endswith(".pdf"):
                pdf_files.append({
                    "name": filename,
                    "data_id": filename.split(".")[0],
                    "path": file_full_path
                })

        if not pdf_files:
            print(f"错误：文件夹 {folder_path} 中未找到PDF文件")
            return None

        print(f"找到 {len(pdf_files)} 个PDF文件")

        # 2. 构造请求数据
        data = {
            "files": [{"name": f["name"], "data_id": f["data_id"]} for f in pdf_files],
            "model_version": model_version
        }
        print(f"准备上传 {len(pdf_files)} 个PDF文件")

        # 3. 申请上传URL
        response = requests.post(upload_url_api, headers=header, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        print("申请上传URL成功")

        # 4. 检查接口返回状态
        if result.get("code") != 0:
            error_msg = result.get("message", "未知错误")
            print(f"申请上传URL失败，原因：{error_msg}")
            return None

        # 5. 提取batch_id和上传URL
        batch_id = result["data"]["batch_id"]
        upload_urls = result["data"]["file_urls"]

        if len(upload_urls) != len(pdf_files):
            print(f"错误：申请到的上传URL数量（{len(upload_urls)}）与文件数量（{len(pdf_files)}）不匹配")
            return None

        print(f"Batch ID: {batch_id}")

        # 6. 逐个上传文件
        uploaded_files = []
        for idx, (file_info, upload_url) in enumerate(zip(pdf_files, upload_urls)):
            file_path = file_info["path"]
            filename = file_info["name"]
            try:
                with open(file_path, 'rb') as f:
                    res_upload = requests.put(upload_url, data=f, timeout=60)
                    res_upload.raise_for_status()

                print(f"✅ 成功上传文件 {filename}")
                uploaded_files.append({
                    "name": filename,
                    "data_id": file_info["data_id"],
                    "upload_status": "success"
                })
            except Exception as upload_err:
                print(f"❌ 文件 {filename} 上传失败，原因：{str(upload_err)}")
                uploaded_files.append({
                    "name": filename,
                    "data_id": file_info["data_id"],
                    "upload_status": "failed",
                    "error": str(upload_err)
                })

        return {
            "batch_id": batch_id,
            "files": uploaded_files
        }

    except requests.exceptions.RequestException as req_err:
        print(f"网络请求错误：{str(req_err)}")
    except KeyError as key_err:
        print(f"响应字段缺失错误：缺少字段 {str(key_err)}，请检查接口返回格式")
    except PermissionError:
        print(f"权限错误：无法读取文件夹或文件，请检查路径权限")
    except Exception as err:
        print(f"未知错误：{str(err)}")
    
    return None

def wait_for_completion(batch_id, check_interval=5):
    """
    轮询检查任务完成状态
    
    Args:
        batch_id (str): 批处理ID
        check_interval (int): 检查间隔（秒）
        
    Returns:
        dict: 最终处理结果
    """
    print(f"开始轮询检查任务状态，Batch ID: {batch_id}")
    attempt = 1
    
    while True:
        print(f"第 {attempt} 次检查任务状态...")
        result = get_batch_results(batch_id)
        
        if not result:
            print("获取结果失败，5秒后重试...")
            time.sleep(check_interval)
            attempt += 1
            continue
            
        # 检查任务状态
        code = result.get("code")
        message = result.get("message", "")
        
        if code == 0:
            # 检查是否有处理结果
            data = result.get("data", {})
            extract_result = data.get("extract_result")
            
            if extract_result is not None and isinstance(extract_result, list):
                # 检查所有文件的状态
                all_completed = True
                has_zip_url = False
                
                for item in extract_result:
                    state = item.get("state", "")
                    # 如果有任何文件仍在处理中，则任务未完成
                    if state in ["pending", "processing", "waiting-file"]:
                        all_completed = False
                    # 检查是否有文件已经完成并有zip链接
                    if "full_zip_url" in item:
                        has_zip_url = True
                
                # 如果所有文件都已完成并且至少有一个有zip链接，则任务真正完成
                if all_completed and has_zip_url:
                    print("任务已完成，获取到处理结果")
                    return result
                elif has_zip_url:
                    # 部分完成也返回，但提示用户
                    print("任务部分完成，获取到部分处理结果")
                    return result
                else:
                    print("任务仍在处理中，5秒后重试...")
            else:
                print("任务仍在处理中，5秒后重试...")
        else:
            print(f"任务处理出错: {message}")
            return result
            
        time.sleep(check_interval)
        attempt += 1

def get_batch_results(batch_id):
    """
    获取批处理结果
    
    Args:
        batch_id (str): 批处理ID
        
    Returns:
        dict: 处理结果
    """
    try:
        url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        response = requests.get(url, headers=header)
        response.raise_for_status()
        
        result = response.json()
        return result
        
    except requests.exceptions.RequestException as req_err:
        print(f"网络请求错误：{str(req_err)}")
    except Exception as err:
        print(f"获取结果时发生错误：{str(err)}")
    
    return None

def download_file(url, local_filename):
    """下载文件"""
    print(f"正在下载: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"下载完成: {local_filename}")
    return local_filename

def extract_zip(zip_path, extract_to):
    """解压zip文件"""
    print(f"正在解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"解压完成: {extract_to}")

def process_batch_result(json_file_path):
    """处理批处理结果文件，下载并解压所有markdown文件"""
    # 读取结果文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    # 获取目录信息
    json_dir = os.path.dirname(json_file_path)
    download_dir = os.path.join(json_dir, "downloads")
    extract_dir = os.path.join(json_dir, "extracted")
    
    # 创建必要的目录
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    # 处理每个文件
    processed_count = 0
    for item in result_data["data"]["extract_result"]:
        data_id = item["data_id"]
        file_name = item["file_name"]
        pdf_name = file_name.replace(".pdf", "")
        
        # 检查文件状态
        state = item.get("state", "")
        if state in ["pending", "processing", "waiting-file"]:
            print(f"警告：文件 {file_name} 状态为 '{state}'，尚未处理完成，跳过处理")
            continue
            
        # 检查full_zip_url是否存在
        if "full_zip_url" not in item:
            print(f"警告：文件 {file_name} 缺少full_zip_url字段，跳过处理")
            continue
            
        zip_url = item["full_zip_url"]
        
        # 下载zip文件
        zip_filename = f"{data_id}.zip"
        zip_path = os.path.join(download_dir, zip_filename)
        try:
            download_file(zip_url, zip_path)
        except Exception as e:
            print(f"下载 {zip_url} 失败: {e}")
            continue
        
        # 解压zip文件到临时目录
        temp_extract_dir = os.path.join(extract_dir, data_id)
        os.makedirs(temp_extract_dir, exist_ok=True)
        try:
            extract_zip(zip_path, temp_extract_dir)
        except Exception as e:
            print(f"解压 {zip_path} 失败: {e}")
            # 清理临时目录
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            continue
        
        # 查找markdown文件
        md_files = list(Path(temp_extract_dir).rglob("*.md"))
        if not md_files:
            print(f"在 {temp_extract_dir} 中未找到markdown文件")
            # 清理临时目录
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            continue
            
        # 重命名并移动markdown文件
        md_file = md_files[0]  # 假设只有一个markdown文件
        new_md_name = f"{pdf_name}.md"
        new_md_path = os.path.join(extract_dir, new_md_name)
        
        # 移动并重命名文件
        shutil.move(str(md_file), new_md_path)
        print(f"已重命名并移动: {md_file.name} -> {new_md_name}")
        
        # 清理临时目录
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        processed_count += 1
    
    print(f"共处理了 {processed_count} 个文件，结果保存在: {extract_dir}")
    return extract_dir

def main():
    print("开始PDF文件批量处理流程...")
    
    # 设置PDF文件夹路径
    folder_path = input("请输入包含PDF文件的文件夹路径: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"错误：路径 {folder_path} 不存在")
        return
    
    if not os.path.isdir(folder_path):
        print(f"错误：{folder_path} 不是一个有效的文件夹")
        return

    # 批量上传PDF文件
    print("\n=== 步骤1：批量上传PDF文件 ===")
    upload_result = batch_upload_pdfs(folder_path)
    
    if not upload_result:
        print("批量上传失败，流程终止")
        return
        
    batch_id = upload_result["batch_id"]
    print(f"\n所有文件已提交上传，Batch ID: {batch_id}")
    
    # 等待服务器处理完成
    print("\n=== 步骤2：等待服务器处理 ===")
    print("现在将每5秒轮询一次服务器，直到任务完成...")
    # 使用轮询方式检查任务完成状态
    result = wait_for_completion(batch_id, check_interval=5)
    
    if not result:
        print("获取处理结果失败，流程终止")
        return
    
    print("\n处理结果获取成功!")
    
    # 保存结果到文件
    result_file = f"batch_result_{batch_id}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到文件: {result_file}")
    
    # 自动处理下载和解压
    print("\n=== 步骤3：下载并解压markdown文件 ===")
    try:
        result_dir = process_batch_result(result_file)
        print(f"\n所有文件处理完成!")
        print(f"最终结果保存在: {result_dir}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()