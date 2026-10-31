#!/usr/bin/env python3
"""Print CUDA/PyTorch diagnostics before starting the RTX 5060 workload."""
import json, platform, subprocess, sys
import torch

info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    info.update({"gpu": p.name, "compute_capability": f"{p.major}.{p.minor}",
                 "total_vram_gb": round(p.total_memory / 1024**3, 2),
                 "bf16_supported": torch.cuda.is_bf16_supported()})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Tiny alloc confirms allocator is functional.
    x = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    info["fp16_matmul"] = "ok" if torch.isfinite(y).all().item() else "failed"
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=5)
    info["nvidia_smi"] = r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
except Exception as exc:
    info["nvidia_smi"] = str(exc)
print(json.dumps(info, indent=2, ensure_ascii=False))
