"""CUDA memory helpers."""

import gc
from typing import Any

import torch


def _resolve_cuda_device(device: Any):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        if device.isdigit():
            return torch.device(f"cuda:{device}")
        return torch.device(device)
    raise TypeError(f"Unsupported CUDA device: {device!r}")


def preallocate_cuda_memory(
    device: Any = 0,
    fraction: float = 0.70,
    leave_free_mb: int = 1024,
    chunk_mb: int = 256,
    clear_cache_first: bool = False,
    verbose: bool = True,
):
    """Reserve most free CUDA memory in PyTorch's caching allocator."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = _resolve_cuda_device(device)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}.")

    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    if leave_free_mb < 0:
        raise ValueError("leave_free_mb must be non-negative.")
    if chunk_mb <= 0:
        raise ValueError("chunk_mb must be positive.")

    torch.cuda.set_device(device)

    if clear_cache_first:
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    leave_free_bytes = int(leave_free_mb * 1024 ** 2)
    chunk_bytes = int(chunk_mb * 1024 ** 2)
    target_bytes = max(0, int((free_bytes - leave_free_bytes) * fraction))

    blocks = []
    allocated_bytes = 0
    try:
        while allocated_bytes < target_bytes:
            this_chunk = min(chunk_bytes, target_bytes - allocated_bytes)
            blocks.append(torch.empty(this_chunk, dtype=torch.uint8, device=device))
            allocated_bytes += this_chunk
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if verbose:
            print(
                "[preallocate_cuda_memory] OOM reached early, "
                f"allocated {allocated_bytes / 1024 ** 2:.1f} MB.",
                flush=True,
            )

    del blocks
    gc.collect()
    torch.cuda.synchronize(device)

    stats = {
        "device": str(device),
        "total_mb": total_bytes / 1024 ** 2,
        "free_before_mb": free_bytes / 1024 ** 2,
        "preallocated_mb": allocated_bytes / 1024 ** 2,
        "torch_allocated_mb": torch.cuda.memory_allocated(device) / 1024 ** 2,
        "torch_reserved_mb": torch.cuda.memory_reserved(device) / 1024 ** 2,
    }

    if verbose:
        print("[preallocate_cuda_memory]", flush=True)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}", flush=True)
            else:
                print(f"  {key}: {value}", flush=True)

    return stats
