import torch
import time
import psutil
import os


# ===================== GPU 测试部分 =====================

def measure_gpu_inference_time(model, input_tensor, num_runs=100, device='cuda'):
    """计算平均 GPU 推理时间 (ms)"""
    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    # GPU 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


def measure_gpu_memory_usage(model, input_tensor, device='cuda'):
    """计算单次 GPU 推理的峰值显存占用 (MB)"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()

    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    torch.cuda.empty_cache()
    return mem_allocated


# ===================== CPU 测试部分 =====================

def measure_cpu_inference_time(model, input_tensor, num_runs=50, device='cpu'):
    """计算平均 CPU 推理时间 (ms)"""
    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end = time.perf_counter()

    avg_time_ms = (end - start) / num_runs * 1000
    return avg_time_ms


def measure_cpu_memory_usage(model, input_tensor, device='cpu'):
    """计算单次 CPU 推理的峰值内存占用 (MB)"""
    process = psutil.Process(os.getpid())
    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    start_mem = process.memory_info().rss
    with torch.no_grad():
        _ = model(input_tensor)
    end_mem = process.memory_info().rss

    mem_used_mb = (end_mem - start_mem) / (1024 ** 2)
    return max(mem_used_mb, 0.0)  # 防止负数（内存回收导致）


# ===================== 综合测试函数 =====================

def evaluate_efficiency(model, input_shape=(1, 3, 224, 224), num_runs=100):
    """同时获取 GPU 和 CPU 的推理时间与内存占用"""
    dummy_input = torch.randn(input_shape)

    # --- GPU ---
    gpu_time_ms = measure_gpu_inference_time(model, dummy_input, num_runs, 'cuda')
    gpu_mem_mb = measure_gpu_memory_usage(model, dummy_input, 'cuda')

    # # --- CPU ---
    # cpu_time_ms = measure_cpu_inference_time(model, dummy_input, num_runs // 2, 'cpu')
    # cpu_mem_mb = measure_cpu_memory_usage(model, dummy_input, 'cpu')

    print(f"🚀 GPU Inference Time: {gpu_time_ms:.2f} ms")
    print(f"💾 GPU Peak Memory Usage: {gpu_mem_mb:.2f} MB")
    # print(f"🧠 CPU Inference Time: {cpu_time_ms:.2f} ms")
    # print(f"📦 CPU Peak Memory Usage: {cpu_mem_mb:.2f} MB")

    return {
        "gpu_time_ms": gpu_time_ms,
        "gpu_mem_mb": gpu_mem_mb,
        # "cpu_time_ms": cpu_time_ms,
        # "cpu_mem_mb": cpu_mem_mb
    }


# ===================== 示例使用 =====================
if __name__ == "__main__":
    from model import build_model  # 假设有一个函数可以构建你的模型
    model = build_model()
    results = evaluate_efficiency(model, input_shape=(1, 3, 224, 224))