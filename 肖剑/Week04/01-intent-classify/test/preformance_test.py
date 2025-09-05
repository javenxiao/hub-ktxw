import requests
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor


def test_request(url, data):
    """发送单个请求并返回响应时间"""
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=10)
        end_time = time.time()
        return {
            "time": end_time - start_time,
            "status": response.status_code,
            "success": response.status_code == 200
        }
    except Exception as e:
        end_time = time.time()
        return {
            "time": end_time - start_time,
            "status": 0,
            "success": False,
            "error": str(e)
        }


def run_performance_test(url, data, concurrency, total_requests):
    """运行性能测试"""
    print(f"开始性能测试: {concurrency} 并发, 总共 {total_requests} 请求")

    # 使用线程池模拟并发请求
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 准备任务列表
        tasks = [executor.submit(test_request, url, data) for _ in range(total_requests)]

        # 收集结果
        results = [task.result() for task in tasks]

    # 分析结果
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    response_times = [r["time"] for r in successful_requests]

    # 输出结果
    print(f"完成请求: {len(successful_requests)} 成功, {len(failed_requests)} 失败")
    if response_times:
        print(f"平均响应时间: {statistics.mean(response_times):.4f} 秒")
        print(f"最小响应时间: {min(response_times):.4f} 秒")
        print(f"最大响应时间: {max(response_times):.4f} 秒")
        print(f"中位数响应时间: {statistics.median(response_times):.4f} 秒")
        print(f"95% 请求响应时间: {sorted(response_times)[int(len(response_times) * 0.95)]:.4f} 秒")

    if failed_requests:
        print("失败的请求:")
        for i, req in enumerate(failed_requests[:5]):  # 只显示前5个失败请求
            print(f"  请求 {i + 1}: {req.get('error', '未知错误')}")
        if len(failed_requests) > 5:
            print(f"  还有 {len(failed_requests) - 5} 个失败请求未显示")

    print("-" * 50)
    return results


if __name__ == "__main__":
    # 测试配置
    url = "http://127.0.0.1:8000"
    test_data = {"text": "外卖味道很好，配送速度快，包装也很精致！"}

    # 测试不同的并发级别
    concurrency_levels = [1, 5, 10]
    total_requests = 100

    print("开始性能测试")
    print("=" * 50)

    for concurrency in concurrency_levels:
        start_time = time.time()
        results = run_performance_test(url, test_data, concurrency, total_requests)
        total_time = time.time() - start_time
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"每秒请求数: {total_requests / total_time:.2f}")
        print("=" * 50)
        print()