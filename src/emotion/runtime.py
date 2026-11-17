"""Small hardware helpers shared by training and inference (no heavy imports)."""
from __future__ import annotations

import torch


def cpu_supports_bf16() -> bool:
    """True on CPUs with native bfloat16 matrix units (Intel AMX / AVX512-BF16)."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
    except OSError:  # macOS / Windows: just use float32
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def autocast_ctx(enabled: bool):
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled)
