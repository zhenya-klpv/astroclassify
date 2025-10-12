# astroclassify/core/device.py
import torch

def pick_device():
    """
    Выбирает оптимальное устройство для инференса:
      - GPU (если доступен и поддерживает sm_120)
      - другой GPU, если есть
      - CPU — если CUDA недоступна или несовместима
    """
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 12:
                return torch.device("cuda:0")  # RTX 5090, Blackwell
        except Exception:
            pass
        # Попробуем другие GPU (например, A16)
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.get_device_capability(i)
                return torch.device(f"cuda:{i}")
            except Exception:
                continue
    return torch.device("cpu")

