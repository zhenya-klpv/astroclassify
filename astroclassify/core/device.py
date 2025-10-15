# astroclassify/core/device.py
from __future__ import annotations
import logging
try:  # optional dependency
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    _HAS_TORCH = False

log = logging.getLogger(__name__)

if _HAS_TORCH:
    # === Perf tuning (один раз при импорте модуля) ===
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True

def pick_device(prefer: str | None = None) -> torch.device:
    """Выбирает устройство: cuda:0 если доступно, иначе cpu."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def gpu_info() -> dict:
    """Возвращает краткую информацию о GPU для логов и healthcheck."""
    info = {
        "torch": torch.__version__,
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        info.update(
            device_name=d.name,
            capability=torch.cuda.get_device_capability(0),
            total_mem_gb=round(d.total_memory / 1024**3, 2),
        )
    return info

def log_gpu_summary():
    """Печатает краткую сводку о GPU в логах при старте."""
    try:
        log.info("GPU summary: %s", gpu_info())
    except Exception as e:
        log.warning("Failed to collect GPU summary: %s", e)
