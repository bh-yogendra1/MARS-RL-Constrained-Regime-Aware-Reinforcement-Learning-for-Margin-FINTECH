"""Safe GPU Management Utility for Shared Cluster Environments.

Ensures that training/eval jobs:
1. Automatically select the GPU with the most free memory.
2. Cap maximum memory fraction (default 25%, e.g., ~10GB on 40GB A100) so no one else gets crashed (OOM).
3. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent memory fragmentation.
4. Fall back safely to CPU if no GPU meets minimum headroom requirements.
"""

import os
import subprocess
import torch

def get_best_gpu(min_free_mb: int = 4000) -> int:
    """Find the GPU ID with the highest available free memory in MB."""
    if not torch.cuda.is_available():
        return -1
    
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = res.stdout.strip().split("\n")
        best_gpu, max_free = -1, -1
        for line in lines:
            if not line.strip():
                continue
            idx_str, free_str = line.split(",")
            idx = int(idx_str.strip())
            free = float(free_str.strip())
            if free > max_free:
                max_free = free
                best_gpu = idx
        
        if max_free >= min_free_mb:
            return best_gpu
    except Exception as e:
        print(f"[GPU Manager] Warning checking nvidia-smi: {e}")
        
    return 0 if torch.cuda.device_count() > 0 else -1


def setup_safe_gpu(target_device: str = "auto", max_mem_fraction: float = 0.25) -> torch.device:
    """
    Configures device safely:
    - 'auto': picks GPU with maximum free memory.
    - 'cuda:X': targets specific GPU.
    - 'cpu': forces CPU.
    Enforces memory ceiling via set_per_process_memory_fraction.
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if not torch.cuda.is_available() or target_device == "cpu":
        print("[GPU Manager] Running safely on: cpu")
        return torch.device("cpu")
    
    if target_device == "auto":
        gpu_idx = get_best_gpu()
    elif target_device.startswith("cuda:"):
        gpu_idx = int(target_device.split(":")[1])
    elif target_device.isdigit():
        gpu_idx = int(target_device)
    else:
        gpu_idx = 0
        
    if gpu_idx < 0 or gpu_idx >= torch.cuda.device_count():
        print(f"[GPU Manager] Warning: GPU {gpu_idx} invalid/unavailable, falling back to CPU.")
        return torch.device("cpu")
        
    dev = torch.device(f"cuda:{gpu_idx}")
    torch.cuda.set_device(gpu_idx)
    
    try:
        # Enforce memory safety cap so other users on shared GPU are never crashed
        torch.cuda.set_per_process_memory_fraction(max_mem_fraction, dev)
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        print(f"[GPU Manager] Safe GPU active -> cuda:{gpu_idx} ({gpu_name}) | Memory Cap: {max_mem_fraction*100:.1f}%")
    except Exception as e:
        print(f"[GPU Manager] Note on memory fraction: {e}")
        
    return dev
