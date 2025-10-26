from __future__ import annotations

import io
import math
from pathlib import Path
import sys
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def gaussian_frame() -> Callable[..., np.ndarray]:
    """Return a factory that builds deterministic Gaussian PSF frames."""

    def _factory(
        size: int = 128,
        centers: Sequence[Tuple[float, float]] = ((64.0, 64.0),),
        fwhm: Iterable[float] | float = 4.0,
        amplitudes: Iterable[float] | float = 5000.0,
        background: float = 20.0,
        noise: float = 0.0,
    ) -> np.ndarray:
        xs, ys = np.indices((size, size), dtype=np.float64)
        image = np.full((size, size), background, dtype=np.float64)

        def _to_iter(value, count):
            if isinstance(value, (list, tuple)):
                return value
            return [value] * count

        centers = tuple(centers)
        fwhm_vals = _to_iter(fwhm, len(centers))
        amp_vals = _to_iter(amplitudes, len(centers))
        for (cx, cy), fwhm_value, amp in zip(centers, fwhm_vals, amp_vals):
            sigma = float(fwhm_value) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            exp_component = ((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2)
            image += float(amp) * np.exp(-exp_component)

        if noise > 0.0:
            rng = np.random.default_rng(seed=12345)
            image += rng.normal(0.0, noise, size=image.shape)
        return image.astype(np.float64)

    return _factory


@pytest.fixture(scope="session")
def png_bytes_from_array() -> Callable[[np.ndarray], bytes]:
    """Encode a floating-point array into an 8-bit PNG suitable for uploads."""

    def _factory(array: np.ndarray) -> bytes:
        arr = array.astype(np.float32)
        arr -= arr.min()
        max_val = float(arr.max()) or 1.0
        arr = np.clip(arr / max_val * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="L")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    return _factory


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return Path(__file__).parent / "golden"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
