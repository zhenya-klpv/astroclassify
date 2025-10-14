# api/main.py
from __future__ import annotations

import io
import os
import time
import math
import json
import zipfile
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from contextlib import contextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, Response
from PIL import Image, ImageDraw

import numpy as np

# Torch / torchvision
import torch
from torch import nn
from torchvision import models, transforms

# Prometheus
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
)

# Внутренние модули
from astroclassify.core.device import pick_device
from astroclassify.api.photometry import (
    has_real_photometry,
    measure_brightness,
    simple_brightness,
    _to_float_array,   # используем конвертер изображений
    _auto_normalize,   # нормализация
)
from astroclassify.api.io_export import (
    export_photometry,
    build_zip_bundle,
)

# Мягкая зависимость для автодетекта источников и FITS-экспорта
try:
    from photutils.detection import DAOStarFinder  # type: ignore
except Exception:
    DAOStarFinder = None  # если нет photutils.detection — эндпоинт вернёт 501

# Опциональная зависимость для быстрого детектора (SEP)
try:
    import sep  # type: ignore
except Exception:
    sep = None  # если нет, поддержим 'dao' как дефолт


# -----------------------------------------------------------------------------
# Логирование
# -----------------------------------------------------------------------------
logger = logging.getLogger("astroclassify")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Limits / validation constants
# -----------------------------------------------------------------------------
_UPLOAD_ENV = os.environ.get("ASTRO_MAX_UPLOAD_BYTES") or os.environ.get("AC_MAX_UPLOAD_BYTES")
MAX_UPLOAD_BYTES = int(_UPLOAD_ENV or str(64 * 1024 * 1024))  # 64 MB
MAX_IMAGE_PIXELS = int(os.environ.get("ASTRO_MAX_IMAGE_PIXELS", str(80_000_000)))  # 80 MP
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

def _astro_error(status: int, code: str, message: str, hint: Optional[str] = None) -> HTTPException:
    payload: Dict[str, Any] = {"code": code, "message": message}
    if hint:
        payload["hint"] = hint
    return HTTPException(status_code=status, detail=payload)

def _validation_error(message: str, *, hint: Optional[str] = None, code: str = "ASTRO_4001") -> HTTPException:
    return _astro_error(400, code, message, hint)

def _dependency_error(message: str, *, hint: Optional[str] = None, code: str = "ASTRO_5021") -> HTTPException:
    return _astro_error(502, code, message, hint)

def _service_error(message: str, *, hint: Optional[str] = None, code: str = "ASTRO_5031") -> HTTPException:
    return _astro_error(503, code, message, hint)

def _validate_upload_size(data: bytes) -> None:
    if not data:
        raise _validation_error("Uploaded file is empty", hint="Provide a non-empty image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise _validation_error(
            "Uploaded file exceeds size limit",
            hint=f"Reduce file size below {MAX_UPLOAD_BYTES} bytes.",
            code="ASTRO_4002",
        )

def _validate_pixel_limit(arr: np.ndarray) -> None:
    if arr is None or arr.size == 0:
        raise _validation_error("Decoded image has no pixels", hint="Verify the input file contains image data.")
    if arr.ndim < 2:
        raise _validation_error("Decoded image is not 2D", hint="Provide a 2D image array.")
    height = int(arr.shape[-2])
    width = int(arr.shape[-1])
    if height * width > MAX_IMAGE_PIXELS:
        raise _validation_error(
            "Image exceeds maximum pixel count",
            hint=f"Reduce resolution below {MAX_IMAGE_PIXELS} pixels (current {height*width}).",
            code="ASTRO_4003",
        )

def _validate_aperture_triplet(
    r: Optional[float],
    r_in: Optional[float],
    r_out: Optional[float],
    *,
    context: str,
) -> None:
    if r is None:
        if r_in is not None or r_out is not None:
            raise _validation_error(
                "Aperture radius `r` is required when specifying annulus radii",
                hint=f"Provide r>0 for {context} or omit r_in/r_out.",
                code="ASTRO_4004",
            )
        return

    if r <= 0:
        raise _validation_error(
            "Aperture radius `r` must be greater than zero",
            hint=f"Increase r for {context}.",
            code="ASTRO_4004",
        )

    if r_in is not None:
        if r_in <= r:
            raise _validation_error(
                "`r_in` must be greater than `r`",
                hint=f"Ensure r_in > r for {context}.",
                code="ASTRO_4005",
            )
    if r_out is not None:
        if r_in is None:
            raise _validation_error(
                "`r_out` requires specifying `r_in`",
                hint=f"Provide both r_in and r_out for {context}.",
                code="ASTRO_4005",
            )
        if r_out <= r_in:
            raise _validation_error(
                "`r_out` must be greater than `r_in`",
                hint=f"Ensure r_out > r_in for {context}.",
                code="ASTRO_4006",
            )

@contextmanager
def _infer_timer(endpoint: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        INFER_HIST.labels(endpoint).observe(time.perf_counter() - start)

# -----------------------------------------------------------------------------
# FastAPI app — создаём сразу!
# -----------------------------------------------------------------------------
app = FastAPI(title="AstroClassify API", version="0.5.0")

@app.exception_handler(HTTPException)
async def _astro_http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    payload: Dict[str, Any]
    if isinstance(detail, dict):
        payload = detail.copy()
        message = payload.get("message", "Request failed")
    else:
        message = str(detail) if detail else "Request failed"
        payload = {"message": message}

    if "code" not in payload:
        if 400 <= exc.status_code < 500:
            payload["code"] = "ASTRO_4001"
        elif exc.status_code == 502:
            payload["code"] = "ASTRO_5021"
        elif exc.status_code == 503:
            payload["code"] = "ASTRO_5031"
        else:
            payload["code"] = "ASTRO_5000"

    payload.setdefault("hint", "See message for details.")
    payload["message"] = message
    return JSONResponse(status_code=exc.status_code, content=payload)

# -----------------------------------------------------------------------------
# Prometheus registry (учёт multiprocess)
# -----------------------------------------------------------------------------
def _build_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        multiprocess.MultiProcessCollector(registry)
    return registry

# Глобальный реестр — именно его нужно отдавать из /metrics
PROM_REGISTRY = _build_registry()

REQ_COUNTER = Counter(
    "astro_requests_total",
    "Total requests per endpoint",
    ["endpoint"],
    registry=PROM_REGISTRY,
)

LATENCY_HIST = Histogram(
    "astro_request_latency_seconds",
    "Request latency per endpoint",
    ["endpoint"],
    registry=PROM_REGISTRY,
)

INFER_HIST = Histogram(
    "astro_inference_seconds",
    "Model/detection inference latency per endpoint",
    ["endpoint"],
    registry=PROM_REGISTRY,
)

# Новые метрики под фотометрию и детекции
PHOT_COUNTER = Counter(
    "astro_photometry_requests_total",
    "Photometry requests by mode",
    ["mode"],
    registry=PROM_REGISTRY,
)
SOURCES_COUNTER = Counter(
    "astro_sources_detected_total",
    "Total sources detected by detector",
    ["detector"],
    registry=PROM_REGISTRY,
)

# Pre-register baseline label values for visibility in /metrics
for _endpoint in ("/health", "/metrics", "/classify", "/classify_batch", "/detect_sources", "/detect_auto", "/preview_apertures"):
    REQ_COUNTER.labels(endpoint=_endpoint)
    LATENCY_HIST.labels(endpoint=_endpoint)
    INFER_HIST.labels(endpoint=_endpoint)

for _mode in ("aperture", "simple"):
    PHOT_COUNTER.labels(mode=_mode)

for _det in ("manual", "dao", "sep"):
    SOURCES_COUNTER.labels(detector=_det)

PREVIEW_PNG_SECONDS = Histogram(
    "astro_preview_png_seconds",
    "Time to render preview overlay PNG",
    ["layout"],
    registry=PROM_REGISTRY,
)

PREVIEW_PLOTS_SECONDS = Histogram(
    "astro_preview_plots_seconds",
    "Time to render preview diagnostic plots",
    ["layout"],
    registry=PROM_REGISTRY,
)

def _track(endpoint: str, method: str):
    start = time.perf_counter()

    class _Tracker:
        def ok(self, status: int = 200):
            LATENCY_HIST.labels(endpoint).observe(time.perf_counter() - start)
            REQ_COUNTER.labels(endpoint).inc()

        def fail(self, status: int):
            LATENCY_HIST.labels(endpoint).observe(time.perf_counter() - start)
            REQ_COUNTER.labels(endpoint).inc()

    return _Tracker()

# -----------------------------------------------------------------------------
# Классификатор (ленивая инициализация)
# -----------------------------------------------------------------------------
_device: torch.device | None = None
_model: nn.Module | None = None
_idx_to_label: List[str] | None = None

_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
_IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

def _ensure_model() -> None:
    """Создаёт модель при первом обращении. Если веса недоступны — работает без них."""
    global _device, _model, _idx_to_label
    if _model is not None:
        return

    _device = pick_device()
    logger.info(f"Using device: {_device}")

    weights = None
    try:
        weights = models.ResNet50_Weights.DEFAULT
    except Exception:
        logger.warning("Torchvision weights not available; using uninitialized model.")

    _model = models.resnet50(weights=weights)
    _model.eval()
    _model.to(_device)

    if weights is not None:
        _idx_to_label = list(weights.meta.get("categories", []))
    else:
        _idx_to_label = [f"class_{i}" for i in range(1000)]

def _classify_bytes(
    data: bytes, imagenet_norm: bool = True, topk: int = 5
) -> List[Dict[str, Any]]:
    _ensure_model()
    assert _model is not None and _device is not None and _idx_to_label is not None

    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")

    tfm = _preprocess
    if imagenet_norm:
        tfm = transforms.Compose(list(_preprocess.transforms) + [_IMAGENET_NORM])

    x = tfm(im).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(x)
        probs = logits.softmax(dim=1)

    values, indices = probs.topk(topk, dim=1)
    values = values[0].tolist()
    indices = indices[0].tolist()

    return [
        {"label": _idx_to_label[i] if i < len(_idx_to_label) else f"class_{i}", "prob": float(p)}
        for i, p in zip(indices, values)
    ]

# -----------------------------------------------------------------------------
# Утилиты экспорта и превью
# -----------------------------------------------------------------------------
def _draw_circles(
    base_img: Image.Image,
    positions: Iterable[tuple[float, float]],
    r: float,
    r_in: float | None,
    r_out: float | None,
    line: int = 2,
) -> Image.Image:
    """Рисуем апертуру (красный) и аннулус (зелёный) поверх изображения."""
    im = base_img.convert("RGB")
    draw = ImageDraw.Draw(im)

    def _ellipse(cx: float, cy: float, radius: float, color: tuple[int, int, int]):
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius, cy + radius
        for offset in range(line):  # имитация толщины линии
            draw.ellipse([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

    for (x, y) in positions:
        _ellipse(x, y, r, (255, 32, 32))
        if r_in and r_in > 0:
            _ellipse(x, y, r_in, (32, 220, 32))
        if r_out and r_out > 0:
            _ellipse(x, y, r_out, (32, 220, 32))

    return im

# -----------------------------------------------------------------------------
# Preview helpers
# -----------------------------------------------------------------------------

FIT_EXTENSIONS = {".fits", ".fit", ".fts"}
PREVIEW_PLOT_ORDER = ("radial", "growth", "background", "snr")
PREVIEW_STRETCHES = {"linear", "log", "asinh"}
PREVIEW_LAYOUTS = {"overlay", "panel"}
PLOT_COLORS = [
    (251, 99, 64),
    (66, 135, 245),
    (46, 204, 113),
    (155, 89, 182),
    (241, 196, 15),
    (26, 188, 156),
]
DEFAULT_PERCENTILE_LOW = 1.0
DEFAULT_PERCENTILE_HIGH = 99.0
BACKGROUND_COLOR = (20, 22, 27)
RESAMPLE_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)

try:
    from astropy.io import fits  # type: ignore

    _HAS_ASTROPY = True
except Exception:  # pragma: no cover - soft dependency
    fits = None  # type: ignore
    _HAS_ASTROPY = False


def _parse_plots(plots: str) -> List[str]:
    if not plots:
        return list(PREVIEW_PLOT_ORDER)
    value = plots.strip().lower()
    if value in ("all", ""):
        return list(PREVIEW_PLOT_ORDER)
    if value == "none":
        return []
    modes = []
    for part in value.replace(";", ",").split(","):
        name = part.strip()
        if not name:
            continue
        if name not in PREVIEW_PLOT_ORDER:
            continue
        modes.append(name)
    if not modes:
        return []
    # remove duplicates preserving order defined by PREVIEW_PLOT_ORDER
    ordered = []
    for mode in PREVIEW_PLOT_ORDER:
        if mode in modes and mode not in ordered:
            ordered.append(mode)
    return ordered


def _load_preview_arrays(
    data: bytes,
    filename: Optional[str],
    percentile_low: float,
    percentile_high: float,
    stretch: str,
) -> tuple[Image.Image, np.ndarray]:
    """Декодирует изображение и строит превью с растяжкой."""
    arr: np.ndarray
    if (
        filename
        and os.path.splitext(filename)[1].lower() in FIT_EXTENSIONS
        and _HAS_ASTROPY
        and fits is not None
    ):
        with fits.open(io.BytesIO(data), memmap=False) as hdul:
            hdu = hdul[0]
            if hdu.data is None:
                raise ValueError("FITS file has no primary image data.")
            arr = np.array(hdu.data, dtype=np.float64, copy=False)
    else:
        # используем _to_float_array из photometry для единообразия
        arr = _to_float_array(data)

    if arr.ndim == 3:
        arr_gray = arr.mean(axis=2)
    else:
        arr_gray = arr

    arr_gray = arr_gray.astype(np.float64, copy=False)
    display = _make_display_image(arr_gray, percentile_low, percentile_high, stretch)
    return display, arr_gray


def _make_display_image(
    arr: np.ndarray,
    percentile_low: float,
    percentile_high: float,
    stretch: str,
) -> Image.Image:
    if arr.size == 0:
        return Image.new("RGB", (32, 32), "black")

    finite = np.isfinite(arr)
    if not np.any(finite):
        arr = np.zeros_like(arr, dtype=np.float64)
        finite = np.ones_like(arr, dtype=bool)

    clipped = arr.copy()
    values = arr[finite]
    lo = float(np.percentile(values, percentile_low)) if percentile_low is not None else float(values.min())
    hi = float(np.percentile(values, percentile_high)) if percentile_high is not None else float(values.max())
    if not math.isfinite(lo):
        lo = float(values.min())
    if not math.isfinite(hi):
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0

    clipped = np.clip(clipped, lo, hi)
    scaled = (clipped - lo) / (hi - lo + 1e-12)

    stretch = stretch.lower()
    if stretch == "log":
        scaled = np.log1p(100.0 * scaled) / math.log(101.0)
    elif stretch == "asinh":
        scaled = np.arcsinh(10.0 * scaled) / np.arcsinh(10.0)

    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    data8 = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(data8, mode="L")
    return img.convert("RGB")


def _profile_stats_for_position(
    data: np.ndarray,
    center: Tuple[float, float],
    profile_max_r: float,
    r_in: Optional[float],
    r_out: Optional[float],
    index: int,
) -> Dict[str, Any]:
    h, w = data.shape
    cx, cy = center
    safe_radius = max(profile_max_r, r_out or 0.0, r_in or 0.0) + 2.0
    x0 = max(int(math.floor(cx - safe_radius)), 0)
    x1 = min(int(math.ceil(cx + safe_radius)) + 1, w)
    y0 = max(int(math.floor(cy - safe_radius)), 0)
    y1 = min(int(math.ceil(cy + safe_radius)) + 1, h)

    if x0 >= x1 or y0 >= y1:
        raise ValueError("Position is outside image bounds.")

    sub = data[y0:y1, x0:x1]
    if sub.size == 0:
        raise ValueError("No data around requested position.")

    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)

    max_bin = max(1, int(math.ceil(profile_max_r)))
    ring_bins = np.floor(rr).astype(int)
    mask = rr <= profile_max_r

    valid_bins = ring_bins[mask]
    valid_vals = sub[mask]

    sums = np.bincount(valid_bins, weights=valid_vals, minlength=max_bin + 1)
    counts = np.bincount(valid_bins, minlength=max_bin + 1)

    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    radii = np.arange(means.shape[0], dtype=np.float64) + 0.5
    flux_cumsum = np.cumsum(sums)
    npix_cumsum = np.cumsum(counts)

    bkg_pixels = np.array([], dtype=np.float64)
    if r_in is not None and r_out is not None and r_out > r_in > 0:
        annulus_mask = (rr >= r_in) & (rr <= r_out)
        bkg_pixels = sub[annulus_mask & np.isfinite(sub)]

    if bkg_pixels.size > 0:
        bkg_mean = float(np.mean(bkg_pixels))
        bkg_rms = float(np.std(bkg_pixels, ddof=0))
        bins = min(48, max(8, int(math.sqrt(bkg_pixels.size))))
        hist_counts, hist_edges = np.histogram(bkg_pixels, bins=bins)
    else:
        bkg_mean = 0.0
        bkg_rms = 0.0
        hist_counts = np.zeros(16, dtype=int)
        hist_edges = np.linspace(-1.0, 1.0, num=17)

    flux_sub = flux_cumsum - npix_cumsum * bkg_mean
    denom = np.sqrt(
        np.maximum(flux_cumsum, 0.0) + npix_cumsum * (bkg_rms**2) + 1e-12
    )
    snr = np.divide(
        flux_sub,
        denom,
        out=np.zeros_like(flux_sub),
        where=denom > 0,
    )

    return {
        "label": f"P{index + 1}",
        "center": {"x": float(cx), "y": float(cy)},
        "radii": radii.tolist(),
        "radial_mean": means.tolist(),
        "flux_cumsum": flux_cumsum.tolist(),
        "flux_sub": flux_sub.tolist(),
        "npix_cumsum": npix_cumsum.tolist(),
        "snr": snr.tolist(),
        "background": {
            "hist": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
            "mean": bkg_mean,
            "rms": bkg_rms,
            "count": int(bkg_pixels.size),
        },
    }


def _build_profile_series(
    data: np.ndarray,
    positions: Sequence[Tuple[float, float]],
    profile_max_r: float,
    r_in: Optional[float],
    r_out: Optional[float],
) -> List[Dict[str, Any]]:
    profiles = []
    for idx, pos in enumerate(positions):
        profiles.append(
            _profile_stats_for_position(
                data=data,
                center=pos,
                profile_max_r=profile_max_r,
                r_in=r_in,
                r_out=r_out,
                index=idx,
            )
        )
    return profiles


def _line_plot(
    width: int,
    height: int,
    series: Sequence[Dict[str, Any]],
    title: str,
    y_label: str,
    x_label: str,
) -> Image.Image:
    img = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    margin_left, margin_right = 60, 20
    margin_top, margin_bottom = 40, 45
    plot_w = max(1, width - margin_left - margin_right)
    plot_h = max(1, height - margin_top - margin_bottom)

    # Determine ranges
    x_min, x_max = math.inf, -math.inf
    y_min, y_max = math.inf, -math.inf
    for item in series:
        xs = item.get("x", [])
        ys = item.get("y", [])
        if not xs or not ys:
            continue
        x_min = min(x_min, min(xs))
        x_max = max(x_max, max(xs))
        y_min = min(y_min, min(ys))
        y_max = max(y_max, max(ys))

    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min == x_max:
        x_min, x_max = 0.0, 1.0
    if not math.isfinite(y_min) or not math.isfinite(y_max) or y_min == y_max:
        y_min, y_max = 0.0, 1.0

    def x_to_px(x: float) -> int:
        if x_max == x_min:
            return margin_left
        return int(margin_left + ((x - x_min) / (x_max - x_min)) * plot_w)

    def y_to_px(y: float) -> int:
        if y_max == y_min:
            return height - margin_bottom
        frac = (y - y_min) / (y_max - y_min)
        return int(height - margin_bottom - frac * plot_h)

    # Axes
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=(90, 92, 100),
    )

    # Grid + ticks (5 steps)
    for i in range(6):
        gx = margin_left + int(plot_w * (i / 5))
        draw.line([(gx, margin_top), (gx, margin_top + plot_h)], fill=(45, 47, 52), width=1)
        value = x_min + (x_max - x_min) * (i / 5)
        draw.text((gx - 10, height - margin_bottom + 6), f"{value:.1f}", fill=(200, 200, 200))

        gy = margin_top + int(plot_h * (i / 5))
        draw.line([(margin_left, gy), (margin_left + plot_w, gy)], fill=(45, 47, 52), width=1)
        value_y = y_max - (y_max - y_min) * (i / 5)
        draw.text((5, gy - 6), f"{value_y:.1f}", fill=(200, 200, 200))

    # Series
    for item in series:
        xs = item.get("x", [])
        ys = item.get("y", [])
        color = item.get("color", (220, 220, 220))
        if len(xs) < 2 or len(ys) < 2:
            continue
        points = [(x_to_px(x), y_to_px(y)) for x, y in zip(xs, ys)]
        draw.line(points, fill=tuple(color), width=2)

    draw.text((margin_left, 12), title, fill=(235, 235, 235))
    draw.text((width // 2 - 40, height - margin_bottom + 22), x_label, fill=(210, 210, 210))
    draw.text((10, margin_top - 28), y_label, fill=(210, 210, 210))

    # Legend
    legend_y = margin_top + 8
    for item in series:
        label = item.get("label")
        color = item.get("color", (220, 220, 220))
        if not label:
            continue
        draw.rectangle(
            [width - margin_right - 100, legend_y, width - margin_right - 80, legend_y + 12],
            fill=tuple(color),
        )
        draw.text((width - margin_right - 75, legend_y - 2), label, fill=(220, 220, 220))
        legend_y += 16

    return img


def _histogram_plot(
    width: int,
    height: int,
    background_data: Sequence[Dict[str, Any]],
    title: str,
) -> Image.Image:
    img = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    margin_left, margin_right = 60, 20
    margin_top, margin_bottom = 40, 45
    plot_w = max(1, width - margin_left - margin_right)
    plot_h = max(1, height - margin_top - margin_bottom)
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=(90, 92, 100),
    )

    max_count = 0
    edge_min, edge_max = math.inf, -math.inf
    for item in background_data:
        counts = item.get("hist", [])
        edges = item.get("edges", [])
        if counts and edges:
            max_count = max(max_count, max(counts))
            edge_min = min(edge_min, edges[0])
            edge_max = max(edge_max, edges[-1])

    if not math.isfinite(edge_min) or not math.isfinite(edge_max) or edge_min == edge_max:
        edge_min, edge_max = -1.0, 1.0
    if max_count <= 0:
        max_count = 1

    def x_to_px(x: float) -> int:
        return int(
            margin_left + ((x - edge_min) / (edge_max - edge_min)) * plot_w
        )

    def count_to_px(c: float) -> int:
        frac = c / max_count
        return int(margin_top + plot_h - frac * plot_h)

    # Draw bins
    for idx, item in enumerate(background_data):
        counts = item.get("hist", [])
        edges = item.get("edges", [])
        color = item.get("color", (200, 200, 200))
        label = item.get("label")
        if not counts or not edges:
            continue
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            x0 = x_to_px(left)
            x1 = x_to_px(right)
            y0 = count_to_px(count)
            draw.rectangle([x0, y0, x1, margin_top + plot_h], fill=tuple(color), outline=None)
        # annotate mean/rms
        mean = item.get("mean", 0.0)
        rms = item.get("rms", 0.0)
        draw.text(
            (width - margin_right - 150, margin_top + idx * 16 + 8),
            f"{label}: μ={mean:.2f} σ={rms:.2f}",
            fill=tuple(color),
        )

    draw.text((margin_left, 12), title, fill=(235, 235, 235))
    draw.text((width // 2 - 40, height - margin_bottom + 22), "Background value", fill=(210, 210, 210))
    draw.text((10, margin_top - 28), "Pixels", fill=(210, 210, 210))
    return img


def _compose_panel(
    preview: Image.Image,
    chart_images: Dict[str, Image.Image],
    selected_plots: Sequence[str],
) -> Tuple[Image.Image, Optional[Image.Image]]:
    """Создаёт итоговое изображение для panel layout и отдельную мозаику графиков."""
    chosen = [name for name in selected_plots if name in chart_images]
    if not chosen:
        return preview, None

    charts = [chart_images[name] for name in chosen]
    chart_w, chart_h = charts[0].size
    cols = 2
    rows = int(math.ceil(len(charts) / cols))
    grid_w = cols * chart_w
    grid_h = rows * chart_h

    mosaic = Image.new("RGB", (grid_w, grid_h), BACKGROUND_COLOR)
    for idx, chart in enumerate(charts):
        row = idx // cols
        col = idx % cols
        mosaic.paste(chart, (col * chart_w, row * chart_h))

    preview_copy = preview.copy()
    preview_copy.thumbnail((grid_w, grid_h), RESAMPLE_LANCZOS)
    canvas_w = preview_copy.width + grid_w + 40
    canvas_h = max(preview_copy.height, grid_h) + 40
    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND_COLOR)
    canvas.paste(preview_copy, (20, 20))
    canvas.paste(mosaic, (preview_copy.width + 30, 20))
    return canvas, mosaic


def _compose_overlay(
    preview: Image.Image,
    chart_images: Dict[str, Image.Image],
    selected_plots: Sequence[str],
    max_plots: int = 2,
) -> Tuple[Image.Image, Optional[Image.Image]]:
    chosen = [name for name in selected_plots if name in chart_images][:max_plots]
    if not chosen:
        return preview, None

    width = preview.width
    panel_height = 260
    panel = Image.new("RGB", (width, panel_height), BACKGROUND_COLOR)
    slot_width = width // len(chosen)
    for idx, name in enumerate(chosen):
        chart = chart_images[name]
        target_w = slot_width - 16
        target_h = panel_height - 20
        chart_copy = chart.copy()
        chart_copy.thumbnail((target_w, target_h), RESAMPLE_LANCZOS)
        offset_x = idx * slot_width + 8
        offset_y = (panel_height - chart_copy.height) // 2
        panel.paste(chart_copy, (offset_x, offset_y))

    combined = Image.new("RGB", (width, preview.height + panel_height), BACKGROUND_COLOR)
    combined.paste(preview, (0, 0))
    combined.paste(panel, (0, preview.height))
    return combined, panel


def _add_labels(
    preview: Image.Image,
    positions: Sequence[Tuple[float, float]],
    label_payload: Dict[str, Any],
) -> Image.Image:
    if not positions and not label_payload:
        return preview

    base = preview.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Meta label
    if label_payload:
        text = f"plots: {','.join(label_payload.get('plots', [])) or 'none'} | layout: {label_payload.get('layout')} | positions: {label_payload.get('count_positions')}"
        padding = 6
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w = int(draw.textlength(text)) if hasattr(draw, "textlength") else len(text) * 6
            text_h = 14
        draw.rectangle(
            [padding, padding, padding + text_w + 6, padding + text_h + 6],
            fill=(0, 0, 0, 150),
        )
        draw.text((padding + 3, padding + 3), text, fill=(255, 255, 255, 255))

    # Position labels
    for idx, (x, y) in enumerate(positions):
        label = f"P{idx + 1}"
        draw.rectangle(
            [x + 4, y - 10, x + 46, y + 6],
            fill=(0, 0, 0, 160),
        )
        draw.text((x + 6, y - 10), label, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


def _placeholder_plot(width: int = 320, height: int = 200, text: str = "No plots") -> Image.Image:
    img = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    draw.text((width // 2 - 40, height // 2 - 10), text, fill=(220, 220, 220))
    return img

# -----------------------------------------------------------------------------
# Эндпоинты
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    t = _track("/health", "GET")
    try:
        payload = {"status": "ok"}
        t.ok()
        return payload
    except Exception as exc:
        t.fail(500)
        raise _service_error("health check failed", hint=str(exc))

@app.get("/metrics")
def metrics():
    """Выдаёт текущие метрики Prometheus из глобального реестра."""
    data = generate_latest(PROM_REGISTRY)  # используем глобальный реестр, не создаём новый
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    topk: int = Query(5, ge=1, le=1000),
    imagenet_norm: bool = Query(True),
):
    t = _track("/classify", "POST")
    try:
        data = await file.read()
        _validate_upload_size(data)
        if topk < 1:
            raise _validation_error("Parameter topk must be >= 1", hint="Increase topk to 1 or higher.", code="ASTRO_4004")
        with _infer_timer("/classify"):
            results = _classify_bytes(data, imagenet_norm=imagenet_norm, topk=topk)
        t.ok()
        return {"filename": file.filename, "results": results}
    except Exception as e:
        logger.exception("classify failed")
        t.fail(500)
        raise _service_error("Classification failed", hint=str(e))

@app.post("/classify_batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    topk: int = Query(5, ge=1, le=1000),
    imagenet_norm: bool = Query(True),
):
    t = _track("/classify_batch", "POST")
    if topk < 1:
        t.fail(400)
        raise _validation_error("Parameter topk must be >= 1", hint="Increase topk to 1 or higher.", code="ASTRO_4004")
    out = []
    status = 200
    for f in files:
        try:
            data = await f.read()
            _validate_upload_size(data)
            with _infer_timer("/classify_batch"):
                results = _classify_bytes(data, imagenet_norm, topk)
            out.append({"filename": f.filename, "results": results})
        except Exception as e:
            logger.exception("batch item failed")
            status = 207  # частично успешно
            out.append({"filename": f.filename, "error": str(e)})

    t.ok(status)
    return {"count": len(files), "results": out}

@app.post("/detect_sources")
async def detect_sources(
    file: UploadFile = File(..., description="Image file: JPG/PNG/TIFF/FITS*"),
    xy: List[str] = Query(
        default=[],
        description='Repeatable center coords "x,y". Example: &xy=120.5,80.2&xy=30,40',
    ),
    r: float | None = Query(default=None, description="Aperture radius (px)"),
    r_in: float | None = Query(default=None, description="Annulus inner radius (px)"),
    r_out: float | None = Query(default=None, description="Annulus outer radius (px)"),
    format: str = Query("json", pattern="^(json|csv|fits)$", description="Export format for photometry results"),
    download: bool = Query(False, description="Force attachment Content-Disposition when true"),
    bundle: str = Query("none", pattern="^(none|zip)$", description="Bundle response into ZIP archive"),
    csv_delimiter: str = Query(",", description="CSV delimiter (use '\\t' or 'tab' for tab)"),
    csv_float_fmt: str = Query(".6f", description="Python format spec for CSV floats"),
    json_indent: int = Query(2, ge=0, le=8, description="Indent for JSON export (ignored when compact=true)"),
    json_compact: bool = Query(False, description="Compact JSON export (overrides indent)"),
):
    """
    Измерение яркости источников.
    - Если заданы `xy` и `r` и доступны astropy+photutils → апертурная фотометрия.
    - Иначе → быстрая оценка яркости (simple_brightness ~ [0..1]).
    Поддерживает экспорт таблицы: json|csv|fits (+ ZIP bundle).
    """
    t = _track("/detect_sources", "POST")
    try:
        data = await file.read()
        _validate_upload_size(data)

        _validate_aperture_triplet(r, r_in, r_out, context="/detect_sources")

        # Парсим координаты "x,y"
        positions: List[tuple[float, float]] = []
        for item in xy:
            try:
                sx, sy = item.split(",", 1)
                positions.append((float(sx), float(sy)))
            except Exception:
                t.fail(400)
                raise _validation_error(
                    f"Invalid xy value: {item!r}",
                    hint="Use format x,y with numeric values.",
                    code="ASTRO_4001",
                )

        do_aperture = bool(positions and r is not None and has_real_photometry())

        if do_aperture:
            with _infer_timer("/detect_sources"):
                results = measure_brightness(
                    data, positions=positions, r=float(r), r_in=r_in, r_out=r_out
                )

            # метрики
            PHOT_COUNTER.labels(mode="aperture").inc()
            SOURCES_COUNTER.labels(detector="manual").inc(len(positions))

            format_mode = format.lower()
            bundle_mode = bundle.lower()
            if bundle_mode not in {"none", "zip"}:
                t.fail(400)
                raise _validation_error(
                    f"Unsupported bundle option: {bundle}",
                    hint="Use bundle=none or bundle=zip.",
                    code="ASTRO_4001",
                )

            return_json_payload = (
                format_mode == "json"
                and bundle_mode == "none"
                and not download
                and not json_compact
                and json_indent == 2
            )

            if return_json_payload:
                payload = {
                    "filename": file.filename,
                    "mode": "aperture",
                    "real_photometry": True,
                    "count": len(positions),
                    "results": results,
                }
                t.ok()
                return payload

            try:
                artifact = export_photometry(
                    results,
                    format_mode,
                    csv_delimiter=csv_delimiter,
                    csv_float_fmt=csv_float_fmt,
                    json_indent=None if json_compact else json_indent,
                    json_compact=json_compact,
                )
            except RuntimeError as err:
                t.fail(501)
                raise _dependency_error(str(err), hint="Install astropy to enable FITS export.")
            except ValueError as err:
                t.fail(400)
                raise _validation_error(str(err), hint="Check export parameters.")

            filename_root = os.path.splitext(file.filename or "image")[0]
            disposition = "attachment" if download else "inline"

            metadata = {
                "endpoint": "detect_sources",
                "filename": file.filename,
                "aperture": {"r": r, "r_in": r_in, "r_out": r_out},
                "count": len(results),
                "format": artifact.extension,
            }

            if bundle_mode == "zip":
                zip_bytes = build_zip_bundle(
                    artifact,
                    metadata=metadata,
                    filename_stem=f"{filename_root}.photometry",
                )
                headers = {
                    "Content-Disposition": f'{disposition}; filename="{filename_root}.photometry_bundle.zip"',
                    "X-Astro-Columns": ",".join(artifact.columns),
                }
                t.ok()
                return Response(content=zip_bytes, media_type="application/zip", headers=headers)

            filename_export = f"{filename_root}.photometry.{artifact.extension}"
            headers = {
                "Content-Disposition": f'{disposition}; filename="{filename_export}"',
                "X-Astro-Columns": ",".join(artifact.columns),
            }
            t.ok()
            return Response(content=artifact.content, media_type=artifact.media_type, headers=headers)

        # Фоллбек — для simple режима экспорт таблиц не применим
        if format.lower() != "json":
            t.fail(400)
            raise _validation_error(
                "Export is only available in aperture mode",
                hint="Provide xy positions and r to enable photometry export.",
            )

        val = simple_brightness(data)
        payload = {
            "filename": file.filename,
            "mode": "simple",
            "real_photometry": has_real_photometry(),
            "simple_brightness": val,
        }
        # метрики
        PHOT_COUNTER.labels(mode="simple").inc()
        SOURCES_COUNTER.labels(detector="manual").inc(0)

        t.ok()
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("detect_sources failed")
        t.fail(500)
        raise _service_error("detect_sources failed", hint=str(e))

@app.post("/detect_auto")
async def detect_auto(
    file: UploadFile = File(..., description="Image file: JPG/PNG/TIFF/FITS*"),
    detector: str = Query("dao", pattern="^(dao|sep)$", description="Detector backend"),
    fwhm: float = Query(3.0, ge=1.0, description="Approx. stellar FWHM in pixels"),
    threshold_sigma: float = Query(5.0, ge=0.1, description="Detection threshold in σ"),
    max_sources: int = Query(50, ge=1, le=5000, description="Limit number of returned sources"),
    sep_minarea: int = Query(5, ge=1, le=5000, description="SEP minimum pixels above threshold"),
    sep_filter_kernel: str = Query("3x3", description="SEP filter kernel (3x3|gaussian|none)"),
    sep_deblend_nthresh: int = Query(32, ge=1, le=256, description="SEP deblend thresholds"),
    sep_deblend_cont: float = Query(0.005, ge=0.0, le=1.0, description="SEP deblend contrast"),
    dao_sharplo: float = Query(0.2, ge=-10.0, le=10.0, description="DAOStarFinder sharplo"),
    dao_roundlo: float = Query(-1.0, ge=-10.0, le=0.0, description="DAOStarFinder roundlo"),
    dao_roundhi: float = Query(1.0, ge=0.0, le=10.0, description="DAOStarFinder roundhi"),
    r: float = Query(5.0, ge=1.0, description="Aperture radius (px)"),
    r_in: float | None = Query(8.0, ge=0.0, description="Annulus inner radius (px)"),
    r_out: float | None = Query(12.0, ge=0.0, description="Annulus outer radius (px)"),
    format: str = Query("json", pattern="^(json|csv|fits)$", description="Export format for photometry results"),
    download: bool = Query(False, description="Force attachment Content-Disposition when true"),
    bundle: str = Query("none", pattern="^(none|zip)$", description="Bundle response into ZIP archive"),
    csv_delimiter: str = Query(",", description="CSV delimiter (use '\\t' or 'tab' for tab)"),
    csv_float_fmt: str = Query(".6f", description="Python format spec for CSV floats"),
    json_indent: int = Query(2, ge=0, le=8, description="Indent for JSON export (ignored when compact=true)"),
    json_compact: bool = Query(False, description="Compact JSON export (overrides indent)"),
):
    """
    Автопоиск источников (DAOStarFinder или SEP) + апертурная фотометрия по найденным центрам.
    Поддерживает экспорт результатов фотометрии: json|csv|fits (+ ZIP bundle).
    """
    t = _track("/detect_auto", "POST")

    if not has_real_photometry():
        t.fail(502)
        raise _dependency_error("Real photometry (photutils/astropy) is required.")

    if detector == "dao" and DAOStarFinder is None:
        t.fail(502)
        raise _dependency_error("DAOStarFinder requires photutils.detection")

    _validate_aperture_triplet(r, r_in, r_out, context="/detect_auto")

    try:
        data = await file.read()
        _validate_upload_size(data)

        # Подготавливаем изображение (градации серого + нормализация)
        arr = _to_float_array(data)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr = _auto_normalize(arr)
        _validate_pixel_limit(arr)

        # Робастная оценка σ фона (MAD → σ)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        robust_sigma = 1.4826 * mad if mad > 0 else float(np.std(arr))
        threshold = threshold_sigma * robust_sigma

        positions: list[tuple[float, float]] = []
        phot: List[Any] = []

        with _infer_timer("/detect_auto"):
            positions = []

            if detector == "sep":
                if sep is None:
                    t.fail(502)
                    raise _dependency_error("SEP is not installed; pip install sep")

                data32 = np.ascontiguousarray(arr.astype(np.float32))
                bkg = sep.Background(data32)
                data_sub = data32 - bkg.back()
                thresh_abs = threshold_sigma * bkg.globalrms
                kernel_name = sep_filter_kernel.strip().lower()
                kernel = None
                if kernel_name in ("3x3", "box", "box3", "top-hat", "tophat"):
                    kernel = np.ones((3, 3), dtype=np.uint8)
                elif kernel_name in ("gaussian", "gauss", "gauss3", "gauss_3x3"):
                    try:
                        kernel = sep.filter.get_filter_kernel("gauss_3x3")  # type: ignore[attr-defined]
                    except Exception:
                        kernel = None
                elif kernel_name in ("none", "off", "0", "false"):
                    kernel = None
                else:
                    raise _validation_error(
                        f"Unsupported sep_filter_kernel: {sep_filter_kernel}",
                        hint="Use 3x3, gaussian, or none.",
                    )

                try:
                    objects = sep.extract(
                        data_sub,
                        thresh_abs,
                        minarea=int(sep_minarea),
                        filter_kernel=kernel,
                        deblend_nthresh=int(sep_deblend_nthresh),
                        deblend_cont=float(sep_deblend_cont),
                    )
                except Exception as err:
                    raise _validation_error(f"SEP extraction failed: {err}")
                if objects is not None and len(objects) > 0:
                    order = np.argsort(objects["flux"])[::-1]
                    for idx in order[:max_sources]:
                        positions.append((float(objects["x"][idx]), float(objects["y"][idx])))
            else:
                try:
                    finder = DAOStarFinder(
                        fwhm=fwhm,
                        threshold=threshold,
                        sharplo=dao_sharplo,
                        sharphi=1.0,
                        roundlo=dao_roundlo,
                        roundhi=dao_roundhi,
                    )
                except Exception as err:
                    raise _validation_error(f"DAOStarFinder init failed: {err}")
                tbl = finder(arr - med)
                if tbl is not None and len(tbl) > 0:
                    order = np.argsort(np.array(tbl["flux"]))[::-1]
                    for idx in order[:max_sources]:
                        x = float(tbl["xcentroid"][idx])
                        y = float(tbl["ycentroid"][idx])
                        positions.append((x, y))

            if not positions:
                PHOT_COUNTER.labels(mode="aperture").inc()
                SOURCES_COUNTER.labels(detector=detector).inc(0)

                t.ok(200)
                return {
                    "filename": file.filename,
                    "mode": "auto-none",
                    "detector": detector,
                    "real_photometry": True,
                    "count": 0,
                    "results": [],
                    "meta": {
                        "fwhm": fwhm,
                        "threshold_sigma": threshold_sigma,
                        "threshold_abs": threshold,
                        "robust_sigma": robust_sigma,
                    },
                }

            phot = measure_brightness(
                data, positions=positions, r=float(r), r_in=r_in, r_out=r_out
            )

        # метрики
        PHOT_COUNTER.labels(mode="aperture").inc()
        SOURCES_COUNTER.labels(detector=detector).inc(len(positions))

        # Экспорт таблицы, если требуется
        format_mode = format.lower()
        bundle_mode = bundle.lower()
        if bundle_mode not in {"none", "zip"}:
            t.fail(400)
            raise _validation_error(
                f"Unsupported bundle option: {bundle}",
                hint="Use bundle=none or bundle=zip.",
            )

        return_json_payload = (
            format_mode == "json"
            and bundle_mode == "none"
            and not download
            and not json_compact
            and json_indent == 2
        )

        if return_json_payload:
            t.ok()
            return {
                "filename": file.filename,
                "mode": "auto-aperture",
                "detector": detector,
                "real_photometry": True,
                "count": len(positions),
                "positions": [{"x": x, "y": y} for x, y in positions],
                "results": phot,
                "meta": {
                    "fwhm": fwhm,
                    "threshold_sigma": threshold_sigma,
                    "threshold_abs": threshold,
                    "robust_sigma": robust_sigma,
                    "sep_minarea": sep_minarea if detector == "sep" else None,
                    "sep_filter_kernel": sep_filter_kernel if detector == "sep" else None,
                    "sep_deblend_nthresh": sep_deblend_nthresh if detector == "sep" else None,
                    "sep_deblend_cont": sep_deblend_cont if detector == "sep" else None,
                    "dao_sharplo": dao_sharplo if detector == "dao" else None,
                    "dao_roundlo": dao_roundlo if detector == "dao" else None,
                    "dao_roundhi": dao_roundhi if detector == "dao" else None,
                },
            }

        try:
            artifact = export_photometry(
                phot,
                format_mode,
                csv_delimiter=csv_delimiter,
                csv_float_fmt=csv_float_fmt,
                json_indent=None if json_compact else json_indent,
                json_compact=json_compact,
            )
        except RuntimeError as err:
            t.fail(502)
            raise _dependency_error(str(err), hint="Install astropy to enable FITS export.")
        except ValueError as err:
            t.fail(400)
            raise _validation_error(str(err), hint="Check export parameters.")

        filename_root = os.path.splitext(file.filename or "image")[0]
        disposition = "attachment" if download else "inline"

        metadata = {
            "endpoint": "detect_auto",
            "filename": file.filename,
            "detector": detector,
            "aperture": {"r": r, "r_in": r_in, "r_out": r_out},
            "count": len(positions),
            "format": artifact.extension,
            "meta": {
                "fwhm": fwhm,
                "threshold_sigma": threshold_sigma,
                "threshold_abs": threshold,
                "robust_sigma": robust_sigma,
                "sep_minarea": sep_minarea if detector == "sep" else None,
                "sep_filter_kernel": sep_filter_kernel if detector == "sep" else None,
                "sep_deblend_nthresh": sep_deblend_nthresh if detector == "sep" else None,
                "sep_deblend_cont": sep_deblend_cont if detector == "sep" else None,
                "dao_sharplo": dao_sharplo if detector == "dao" else None,
                "dao_roundlo": dao_roundlo if detector == "dao" else None,
                "dao_roundhi": dao_roundhi if detector == "dao" else None,
            },
        }

        if bundle_mode == "zip":
            zip_bytes = build_zip_bundle(
                artifact,
                metadata=metadata,
                filename_stem=f"{filename_root}.auto.photometry",
            )
            headers = {
                "Content-Disposition": f'{disposition}; filename="{filename_root}.auto.photometry_bundle.zip"',
                "X-Astro-Columns": ",".join(artifact.columns),
            }
            t.ok()
            return Response(content=zip_bytes, media_type="application/zip", headers=headers)

        headers = {
            "Content-Disposition": f'{disposition}; filename="{filename_root}.auto.photometry.{artifact.extension}"',
            "X-Astro-Columns": ",".join(artifact.columns),
        }
        t.ok()
        return Response(content=artifact.content, media_type=artifact.media_type, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("detect_auto failed")
        t.fail(500)
        raise _service_error("detect_auto failed", hint=str(e))

@app.post("/preview_apertures")
async def preview_apertures(
    file: UploadFile = File(..., description="Image file: JPG/PNG/TIFF/FITS*"),
    xy: List[str] = Query(
        default=[],
        description='Repeatable center coords "x,y". Example: &xy=120.5,80.2&xy=30,40',
    ),
    r: float = Query(5.0, ge=1.0, description="Aperture radius (px)"),
    r_in: float | None = Query(8.0, ge=0.0, description="Annulus inner radius (px)"),
    r_out: float | None = Query(12.0, ge=0.0, description="Annulus outer radius (px)"),
    line: int = Query(2, ge=1, le=8, description="Outline thickness"),
    plots: str = Query("all", description="Diagnostic plots: none|radial|growth|background|snr|all"),
    layout: str = Query("overlay", description="Layout: overlay or panel"),
    bundle: str = Query("png", description="Response bundle: png or zip"),
    profile_max_r: float = Query(20.0, ge=1.0, le=256.0, description="Max radius for profiles (px)"),
    percentile_low: float = Query(DEFAULT_PERCENTILE_LOW, ge=0.0, le=100.0, description="Lower percentile for display stretch"),
    percentile_high: float = Query(DEFAULT_PERCENTILE_HIGH, ge=0.0, le=100.0, description="Upper percentile for display stretch"),
    stretch: str = Query("linear", description="Display stretch: linear|log|asinh"),
    labels: bool = Query(True, description="Render overlay labels"),
):
    """
    Расширенный превью-рендер апертур:
    - Overlay layout: базовое изображение + контуры + мини-панель графиков (до 2).
    - Panel layout: составной PNG с превью и 2–4 графиками.
    Параметр bundle=zip вернёт архив (preview.png, plots.png, metrics.json).
    """
    t = _track("/preview_apertures", "POST")
    try:
        data = await file.read()
        _validate_upload_size(data)

        _validate_aperture_triplet(r, r_in, r_out, context="/preview_apertures")

        # Парсим координаты
        positions: List[tuple[float, float]] = []
        for item in xy:
            try:
                sx, sy = item.split(",", 1)
                positions.append((float(sx), float(sy)))
            except Exception:
                t.fail(400)
                raise _validation_error(
                    f"Invalid xy value: {item!r}",
                    hint="Use format x,y with numeric values.",
                )

        if not positions:
            t.fail(400)
            raise _validation_error(
                "At least one xy coordinate is required",
                hint="Pass xy parameters like xy=120.5,80.2.",
            )

        if percentile_low >= percentile_high:
            t.fail(400)
            raise _validation_error(
                "percentile_low must be less than percentile_high",
                hint="Adjust percentile range.",
            )

        layout_mode = layout.lower()
        if layout_mode not in PREVIEW_LAYOUTS:
            t.fail(400)
            raise _validation_error(
                f"Unsupported layout: {layout}",
                hint="Use layout=overlay or layout=panel.",
            )

        bundle_mode = bundle.lower()
        if bundle_mode not in {"png", "zip"}:
            t.fail(400)
            raise _validation_error(
                f"Unsupported bundle option: {bundle}",
                hint="Use bundle=png or bundle=zip.",
            )

        stretch_mode = stretch.lower()
        if stretch_mode not in PREVIEW_STRETCHES:
            t.fail(400)
            raise _validation_error(
                f"Unsupported stretch: {stretch}",
                hint="Use stretch=linear|log|asinh.",
            )

        selected_plots = _parse_plots(plots)

        # Подготовка превью и данных
        if (
            file.filename
            and os.path.splitext(file.filename)[1].lower() in FIT_EXTENSIONS
            and not _HAS_ASTROPY
        ):
            t.fail(502)
            raise _dependency_error("FITS preview requires astropy to be installed")

        png_start = time.perf_counter()
        preview_img, array_data = _load_preview_arrays(
            data=data,
            filename=file.filename,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            stretch=stretch_mode,
        )
        _validate_pixel_limit(array_data)
        overlay = _draw_circles(preview_img, positions, r=r, r_in=r_in, r_out=r_out, line=line)

        profile_start = time.perf_counter()
        profiles: List[Dict[str, Any]] = []
        if selected_plots:
            try:
                profiles = _build_profile_series(
                    array_data, positions, profile_max_r, r_in=r_in, r_out=r_out
                )
            except ValueError as exc:
                t.fail(400)
                raise _validation_error(str(exc))
        chart_images: Dict[str, Image.Image] = {}
        if profiles and selected_plots:
            for idx, item in enumerate(profiles):
                item["color"] = PLOT_COLORS[idx % len(PLOT_COLORS)]

            def _series_for(field: str) -> List[Dict[str, Any]]:
                return [
                    {
                        "x": profile["radii"],
                        "y": profile[field],
                        "label": profile["label"],
                        "color": profile["color"],
                    }
                    for profile in profiles
                    if profile.get(field)
                ]

            if "radial" in selected_plots:
                chart_images["radial"] = _line_plot(
                    width=520,
                    height=360,
                    series=_series_for("radial_mean"),
                    title="Radial profile",
                    y_label="Mean intensity",
                    x_label="Radius (px)",
                )

            if "growth" in selected_plots:
                chart_images["growth"] = _line_plot(
                    width=520,
                    height=360,
                    series=[
                        {
                            "x": profile["radii"],
                            "y": profile["flux_sub"],
                            "label": profile["label"],
                            "color": profile["color"],
                        }
                        for profile in profiles
                    ],
                    title="Growth curve",
                    y_label="Flux (bg-sub)",
                    x_label="Radius (px)",
                )

            if "snr" in selected_plots:
                chart_images["snr"] = _line_plot(
                    width=520,
                    height=360,
                    series=[
                        {
                            "x": profile["radii"],
                            "y": profile["snr"],
                            "label": profile["label"],
                            "color": profile["color"],
                        }
                        for profile in profiles
                    ],
                    title="SNR vs Radius",
                    y_label="SNR",
                    x_label="Radius (px)",
                )

            if "background" in selected_plots:
                chart_images["background"] = _histogram_plot(
                    width=520,
                    height=360,
                    background_data=[
                        {
                            "hist": profile["background"]["hist"],
                            "edges": profile["background"]["edges"],
                            "mean": profile["background"]["mean"],
                            "rms": profile["background"]["rms"],
                            "label": profile["label"],
                            "color": profile["color"],
                        }
                        for profile in profiles
                    ],
                    title="Background histogram",
                )

        png_seconds = time.perf_counter() - png_start
        plots_seconds = max(0.0, time.perf_counter() - profile_start) if chart_images else 0.0

        PREVIEW_PNG_SECONDS.labels(layout_mode).observe(png_seconds)
        PREVIEW_PLOTS_SECONDS.labels(layout_mode).observe(plots_seconds)

        label_payload = {
            "plots": selected_plots,
            "layout": layout_mode,
            "count_positions": len(positions),
        }
        overlay_with_labels = _add_labels(overlay, positions, label_payload) if labels else overlay

        if layout_mode == "panel":
            final_img, plots_img = _compose_panel(overlay_with_labels, chart_images, selected_plots)
        else:
            final_img, plots_img = _compose_overlay(overlay_with_labels, chart_images, selected_plots)

        filename_root = os.path.splitext(file.filename or "image")[0]
        metrics_payload = {
            "astro_preview_png_seconds": png_seconds,
            "astro_preview_plots_seconds": plots_seconds,
            "labels": label_payload,
        }

        if bundle_mode == "zip":
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                preview_buf = io.BytesIO()
                overlay_with_labels.save(preview_buf, format="PNG")
                zf.writestr("preview.png", preview_buf.getvalue())

                plots_buf = io.BytesIO()
                if plots_img is not None:
                    plots_img.save(plots_buf, format="PNG")
                else:
                    _placeholder_plot().save(plots_buf, format="PNG")
                zf.writestr("plots.png", plots_buf.getvalue())
                zf.writestr("metrics.json", json.dumps(metrics_payload, ensure_ascii=False, indent=2))

            zip_buf.seek(0)
            headers = {
                "Content-Disposition": f'attachment; filename="{filename_root}.preview_bundle.zip"',
                "X-Astro-Preview-Png-Seconds": f"{png_seconds:.4f}",
                "X-Astro-Preview-Plots-Seconds": f"{plots_seconds:.4f}",
            }
            t.ok()
            return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        buf.seek(0)
        headers = {
            "Content-Disposition": f'inline; filename="{filename_root}.preview.png"',
            "X-Astro-Preview-Png-Seconds": f"{png_seconds:.4f}",
            "X-Astro-Preview-Plots-Seconds": f"{plots_seconds:.4f}",
        }
        t.ok()
        return StreamingResponse(buf, media_type="image/png", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("preview_apertures failed")
        t.fail(500)
        raise _service_error("preview_apertures failed", hint=str(e))
