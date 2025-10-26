# api/main.py
from __future__ import annotations

import io
import os
import time
import math
import json
import zipfile
import logging
import tempfile
import re
import requests
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from contextlib import contextmanager

from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, Response
from PIL import Image, ImageDraw

try:  # Pillow >=9.1
    from PIL import ImageDecompressionBombError
except ImportError:  # pragma: no cover - legacy fallback
    ImageDecompressionBombError = Image.DecompressionBombError  # type: ignore[attr-defined]

import numpy as np

# Torch / torchvision (optional)
try:
    import torch
    from torch import nn
    from torchvision import models, transforms

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore
    _HAS_TORCH = False

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
    ImageProbe,
    ImageValidationError,
    probe_image_bytes,
    _to_float_array,   # используем конвертер изображений
    _auto_normalize,   # нормализация
)
from astroclassify.api.io_export import (
    export_photometry,
    build_zip_bundle,
)
from astroclassify.data_sources import (
    CutoutRequest,
    CutoutResult,
    CutoutError,
    get_cutout_provider,
)
from astroclassify.observability import (
    configure_observability,
    ensure_logging_filter,
    get_tracer,
)
from astroclassify.psf_photometry import run_psf_photometry, PSFPhotometryResult, PSFModel

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
ensure_logging_filter()
logger = logging.getLogger("astroclassify")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | trace=%(trace_id)s span=%(span_id)s | %(message)s",
)

# -----------------------------------------------------------------------------
# Limits / validation constants
# -----------------------------------------------------------------------------
_UPLOAD_ENV = os.environ.get("ASTRO_MAX_UPLOAD_BYTES") or os.environ.get("AC_MAX_UPLOAD_BYTES")
MAX_UPLOAD_BYTES = int(_UPLOAD_ENV or str(64 * 1024 * 1024))  # 64 MB
MAX_IMAGE_PIXELS = int(os.environ.get("ASTRO_MAX_IMAGE_PIXELS", str(80_000_000)))  # 80 MP
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
MAX_PIXEL_PER_BYTE = float(os.environ.get("ASTRO_MAX_PIXEL_PER_BYTE", "20000"))

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


def _validate_probe_limits(probe: ImageProbe, data_len: int, *, context: str) -> None:
    if probe.width is not None and probe.height is not None:
        width = int(probe.width)
        height = int(probe.height)
        if width <= 0 or height <= 0:
            raise _validation_error(
                "Image has invalid dimensions",
                hint=f"Detected size {width}x{height} for {context} is not valid.",
                code="ASTRO_4003",
            )
        if width * height > MAX_IMAGE_PIXELS:
            raise _validation_error(
                "Image exceeds maximum pixel count",
                hint=f"Reduce resolution below {MAX_IMAGE_PIXELS} pixels (current {width*height}).",
                code="ASTRO_4003",
            )
        # Heuristic: reject suspicious compression bombs
        if data_len > 0:
            pixels_per_byte = (width * height) / max(1, data_len)
            if pixels_per_byte > MAX_PIXEL_PER_BYTE:
                raise _validation_error(
                    "Image compression ratio is suspiciously high",
                    hint="Re-encode the image with less extreme compression to ensure safety.",
                    code="ASTRO_4007",
                )


def _preflight_image_bytes(data: bytes, filename: Optional[str], *, context: str) -> ImageProbe:
    try:
        probe = probe_image_bytes(data, filename=filename)
    except ImageValidationError as exc:
        raise _validation_error(
            str(exc),
            hint="Upload a valid image file with a supported extension and signature.",
            code="ASTRO_4007",
        ) from exc

    _validate_probe_limits(probe, len(data), context=context)
    return probe


@dataclass
class _CalibrationParams:
    exptime: Optional[float]
    gain: Optional[float]
    zeropoint: Optional[float]
    mag_system: Optional[str]


def _calibrated_magnitude(
    flux: float,
    flux_err: Optional[float],
    params: _CalibrationParams,
) -> Optional[Tuple[float, Optional[float]]]:
    if params.zeropoint is None or flux <= 0:
        return None
    gain = params.gain or 1.0
    counts = flux * gain
    flux_err_counts = flux_err * gain if (flux_err is not None) else None
    if params.exptime and params.exptime > 0:
        counts /= params.exptime
        if flux_err_counts is not None:
            flux_err_counts /= params.exptime
    mag = params.zeropoint - 2.5 * math.log10(counts)
    mag_err = None
    if flux_err_counts is not None and flux_err_counts > 0:
        mag_err = 1.0857362047581296 * (flux_err_counts / counts)
    return float(mag), (float(mag_err) if mag_err is not None else None)


def _merge_psf_results(
    results: List[Dict[str, Any]],
    psf_payload: List[Tuple[int, PSFPhotometryResult]],
    psf_model: Optional[PSFModel],
    calibration: _CalibrationParams,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if psf_model is not None:
        summary = {
            "shape": psf_model.kernel.shape,
            "center": psf_model.center,
            "fwhm_major": psf_model.fwhm_major,
            "fwhm_minor": psf_model.fwhm_minor,
            "ellipticity": psf_model.ellipticity,
            "position_angle": psf_model.position_angle,
        }
    for index, psf in psf_payload:
        if 0 <= index < len(results):
            entry = results[index]
            entry["photometry_mode"] = "psf"
            entry["flux_psf"] = psf.flux_psf
            entry["flux_err_psf"] = psf.flux_err_psf
            entry["snr_psf"] = psf.snr_psf
            entry["chi2_psf"] = psf.chi2_psf
            entry["psf_fwhm_major"] = psf.fwhm_major
            entry["psf_fwhm_minor"] = psf.fwhm_minor
            entry["psf_ellipticity"] = psf.ellipticity
            entry["psf_position_angle"] = psf.position_angle
            entry["psf_background"] = psf.background
            mag_info = _calibrated_magnitude(psf.flux_psf, psf.flux_err_psf, calibration)
            if mag_info is not None:
                mag, mag_err = mag_info
                entry["mag"] = mag
                if mag_err is not None:
                    entry["mag_err"] = mag_err
            if calibration.mag_system:
                entry["mag_system"] = calibration.mag_system.lower()
    return summary


def _annotate_aperture_results(
    results: List[Dict[str, Any]],
    radius: Optional[float],
    calibration: _CalibrationParams,
) -> None:
    ap_area = math.pi * (radius or 0.0) ** 2 if radius else None
    for entry in results:
        entry.setdefault("photometry_mode", "aperture")
        flux = None
        if isinstance(entry, dict):
            for key in ("flux_sub", "aperture_sum", "flux"):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    flux = float(value)
                    if key == "flux_sub" or key == "flux":
                        break
        if flux is None:
            continue
        bkg_mean = float(entry.get("bkg_mean", 0.0)) if isinstance(entry, dict) else 0.0
        bkg_area = float(entry.get("bkg_area", ap_area or 0.0)) if isinstance(entry, dict) else 0.0
        shot = max(flux, 0.0)
        background = max(bkg_mean, 0.0) * bkg_area
        variance = shot + background
        flux_err = math.sqrt(max(variance, 1.0))
        entry["flux"] = flux
        entry["flux_err"] = flux_err
        mag_info = _calibrated_magnitude(flux, flux_err, calibration)
        if mag_info is not None:
            mag, mag_err = mag_info
            entry["mag"] = mag
            if mag_err is not None:
                entry["mag_err"] = mag_err
            if calibration.mag_system:
                entry["mag_system"] = calibration.mag_system.lower()


def _finalise_photometry_results(
    image_array: Optional[np.ndarray],
    results: List[Dict[str, Any]],
    positions: Sequence[Tuple[float, float]],
    *,
    phot_mode: str,
    aperture_radius: Optional[float],
    psf_stamp_radius: int,
    psf_fit_radius: int,
    calibration: _CalibrationParams,
) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    _annotate_aperture_results(results, aperture_radius, calibration)
    psf_success = False
    if phot_mode == "psf" and image_array is not None and positions:
        psf_entries, psf_model = run_psf_photometry(
            image_array,
            positions,
            stamp_radius=psf_stamp_radius,
            fit_radius=psf_fit_radius,
        )
        if psf_entries:
            psf_success = True
            psf_summary = _merge_psf_results(results, psf_entries, psf_model, calibration)
            if psf_summary:
                extras["psf_model"] = psf_summary
    default_mode = "psf" if psf_success else "aperture"
    for entry in results:
        entry.setdefault("photometry_mode", default_mode)

    if calibration.zeropoint is not None:
        cal_dict = {
            "zeropoint": calibration.zeropoint,
            "mag_system": (calibration.mag_system or "AB").lower(),
        }
        if calibration.exptime is not None:
            cal_dict["exptime"] = calibration.exptime
        if calibration.gain is not None:
            cal_dict["gain"] = calibration.gain
        extras["calibration"] = cal_dict
    return extras


def _validate_aperture_triplet(
    r: Optional[float],
    r_in: Optional[float],
    r_out: Optional[float],
    *,
    context: str,
    allow_missing_r: bool = False,
) -> None:
    if r is None:
        if allow_missing_r:
            if r_in is not None and r_in <= 0:
                raise _validation_error(
                    "`r_in` must be greater than zero",
                    hint=f"Adjust r_in for {context}.",
                    code="ASTRO_4005",
                )
            if r_out is not None and (r_in is None or r_out <= r_in):
                raise _validation_error(
                    "`r_out` must be greater than `r_in`",
                    hint=f"Ensure r_out > r_in for {context}.",
                    code="ASTRO_4006",
                )
            return
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


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise _dependency_error(
            "PyTorch/torchvision are required for classification endpoints",
            hint="Install torch and torchvision or disable /v1/classify* routes.",
            code="ASTRO_5022",
        )

# -----------------------------------------------------------------------------
# FastAPI app — создаём сразу!
# -----------------------------------------------------------------------------
app = FastAPI(title="AstroClassify API", version="0.5.0")
API_PREFIX = "/v1"
router = APIRouter(prefix=API_PREFIX)

if not _HAS_TORCH:
    logger.warning('PyTorch is not installed; /v1/classify endpoints will return ASTRO_5022')

configure_observability(app)
TRACER = get_tracer("astroclassify.api")


def _versioned(path: str) -> str:
    if path.startswith(API_PREFIX):
        return path
    return f"{API_PREFIX}{path}"


@app.middleware("http")
async def _version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-AstroClassify-API"] = "1"
    return response

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
    "Total requests per endpoint and status",
    ["endpoint", "status"],
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

BACKGROUND_SECONDS = Histogram(
    "astro_background_seconds",
    "Background estimation time per endpoint",
    ["endpoint"],
    registry=PROM_REGISTRY,
)

ROI_SECONDS = Histogram(
    "astro_roi_seconds",
    "ROI extraction time per endpoint",
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
EXPORT_BYTES_COUNTER = Counter(
    "astro_export_bytes_total",
    "Total bytes produced by photometry/export operations",
    ["format"],
    registry=PROM_REGISTRY,
)

# Pre-register baseline label values for visibility in /metrics
for _endpoint in ("health", "ready", "metrics", "classify", "classify_batch", "detect_sources", "detect_auto", "preview_apertures", "cutout"):
    path = _versioned(f"/{_endpoint}")
    for _status in ("200", "207", "400", "404", "500"):
        REQ_COUNTER.labels(endpoint=path, status=_status)
    LATENCY_HIST.labels(endpoint=path)
    INFER_HIST.labels(endpoint=path)
    BACKGROUND_SECONDS.labels(endpoint=path)
    ROI_SECONDS.labels(endpoint=path)

for _mode in ("aperture", "simple", "psf"):
    PHOT_COUNTER.labels(mode=_mode)

for _det in ("manual", "dao", "sep"):
    SOURCES_COUNTER.labels(detector=_det)

PREVIEW_PNG_SECONDS = Histogram(
    "astro_preview_png_seconds",
    "Time to render preview overlay PNG",
    ["detector", "format", "sensor", "pixels_bin"],
    registry=PROM_REGISTRY,
)

PREVIEW_PLOTS_SECONDS = Histogram(
    "astro_preview_plots_seconds",
    "Time to render preview diagnostic plots",
    ["detector", "format", "sensor", "pixels_bin"],
    registry=PROM_REGISTRY,
)

_LABEL_SANITIZER = re.compile(r"[^0-9a-zA-Z:_\-/\.]+")
_PIXEL_BIN_BOUNDARIES = (
    (0, 524_288, "lt_0_5mp"),          # up to ~0.5 MP
    (524_288, 2_097_152, "0_5_2mp"),   # 0.5-2 MP
    (2_097_152, 8_388_608, "2_8mp"),   # 2-8 MP
    (8_388_608, 20_971_520, "8_20mp"), # 8-20 MP
    (20_971_520, float("inf"), "gt_20mp"),
)

def _sanitize_label(value: Optional[str], default: str) -> str:
    if value is None:
        return default
    trimmed = value.strip()
    if not trimmed:
        return default
    cleaned = _LABEL_SANITIZER.sub("_", trimmed.lower())
    return cleaned or default

def _pixels_bin_from_shape(shape: Optional[Sequence[int]]) -> str:
    if not shape or len(shape) < 2:
        return "unknown"
    height = int(shape[-2])
    width = int(shape[-1])
    if height <= 0 or width <= 0:
        return "unknown"
    pixels = height * width
    for lower, upper, label in _PIXEL_BIN_BOUNDARIES:
        if pixels <= upper:
            return label
    return _PIXEL_BIN_BOUNDARIES[-1][2]

def _sensor_label(
    request: Optional[Request],
    filename: Optional[str],
    array_shape: Optional[Sequence[int]] = None,
) -> str:
    header_val = None
    if request is not None:
        header_val = (
            request.headers.get("x-astro-sensor")
            or request.headers.get("x-sensor")
            or request.headers.get("x-instrument")
        )
    if header_val:
        return _sanitize_label(header_val, "unknown")
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in FIT_EXTENSIONS:
            return "fits"
        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            return "rgb"
    if array_shape is not None:
        if len(array_shape) == 3 and array_shape[0] in (3, 4):
            return "rgb"
        if len(array_shape) >= 2:
            return "mono"
    return "unknown"

def _preview_metric_labels(
    *,
    detector: str,
    response_format: str,
    request: Optional[Request],
    filename: Optional[str],
    array_shape: Optional[Sequence[int]],
) -> Tuple[str, str, str, str]:
    detector_label = _sanitize_label(detector or "unknown", "unknown")
    format_label = _sanitize_label(response_format or "png", "png")
    sensor_label = _sensor_label(request, filename, array_shape)
    pixels_bin = _pixels_bin_from_shape(array_shape)
    return detector_label, format_label, sensor_label, pixels_bin

for _fmt in ("csv", "json", "fits", "zip"):
    EXPORT_BYTES_COUNTER.labels(format=_fmt)

PREVIEW_PNG_SECONDS.labels("manual", "png", "unknown", "unknown")
PREVIEW_PLOTS_SECONDS.labels("manual", "png", "unknown", "unknown")

def _track(endpoint: str, method: str):
    start = time.perf_counter()

    class _Tracker:
        def ok(self, status: int = 200):
            status_str = str(status)
            LATENCY_HIST.labels(endpoint=endpoint).observe(time.perf_counter() - start)
            REQ_COUNTER.labels(endpoint=endpoint, status=status_str).inc()

        def fail(self, status: int):
            status_str = str(status)
            LATENCY_HIST.labels(endpoint=endpoint).observe(time.perf_counter() - start)
            REQ_COUNTER.labels(endpoint=endpoint, status=status_str).inc()

    return _Tracker()


# -----------------------------------------------------------------------------
# Классификатор (ленивая инициализация)
# -----------------------------------------------------------------------------
_device: torch.device | None = None
_model: nn.Module | None = None
_idx_to_label: List[str] | None = None
_MODEL_INFO: Dict[str, Any] = {
    "name": "resnet50",
    "weights": None,
    "device": None,
    "categories": 0,
}

if _HAS_TORCH:
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
else:  # pragma: no cover - torch unavailable
    _preprocess = None
    _IMAGENET_NORM = None

def _ensure_model() -> None:
    """Создаёт модель при первом обращении. Если веса недоступны — работает без них."""
    _require_torch()
    global _device, _model, _idx_to_label, _MODEL_INFO
    if _model is not None:
        return

    assert models is not None and torch is not None

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

    weight_name = getattr(weights, "name", "DEFAULT") if weights is not None else None
    _MODEL_INFO.update(
        {
            "name": "resnet50",
            "weights": weight_name,
            "device": str(_device),
            "categories": len(_idx_to_label or []),
        }
    )

def _classify_bytes(
    data: bytes,
    imagenet_norm: bool = True,
    topk: int = 5,
    filename: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _require_torch()
    assert _preprocess is not None
    _ensure_model()
    assert _model is not None and _device is not None and _idx_to_label is not None

    _preflight_image_bytes(data, filename, context="/classify")

    decode_start = time.perf_counter()
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
    except ImageDecompressionBombError as exc:
        raise ImageValidationError("Image is too large to process safely.") from exc
    except Exception as exc:
        raise ImageValidationError(f"Failed to decode image: {exc}") from exc
    elapsed = time.perf_counter() - decode_start
    if elapsed > _DECODE_TIMEOUT_SECONDS:
        raise ImageValidationError(
            f"Image decoding exceeded {_DECODE_TIMEOUT_SECONDS:.1f}s safety limit."
        )

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
PREVIEW_LAYOUTS = {"overlay", "panel", "grid", "row"}
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
FWHM_FACTOR = 2.354820045
ROI_MIN_MARGIN = 32.0


class _BaseImageSource:
    shape: Tuple[int, int]

    def get_full(self) -> np.ndarray:
        raise NotImplementedError

    def get_roi(self, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _ArrayImageSource(_BaseImageSource):
    def __init__(self, array: np.ndarray):
        arr = np.asarray(array)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        self._array = arr.astype(np.float64, copy=False)
        if self._array.ndim != 2:
            raise ValueError("Expected 2D image array for ArrayImageSource")
        self.shape = (self._array.shape[0], self._array.shape[1])

    def get_full(self) -> np.ndarray:
        return self._array

    def get_roi(self, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        return self._array[y0:y1, x0:x1]


class _FitsImageSource(_BaseImageSource):
    def __init__(self, path: str):
        if not _HAS_ASTROPY or fits is None:
            raise RuntimeError("FITS support requires astropy")
        self._path = path
        self._hdul = fits.open(path, memmap=True, mode="readonly")
        data = self._hdul[0].data
        if data is None:
            self._hdul.close()
            raise ValueError("FITS file contains no primary image data")
        self._data = data
        if self._data.ndim == 2:
            self.shape = (self._data.shape[0], self._data.shape[1])
        elif self._data.ndim == 3:
            self.shape = (self._data.shape[-2], self._data.shape[-1])
        else:
            self._hdul.close()
            raise ValueError("Unsupported FITS data dimensionality")

    def get_full(self) -> np.ndarray:
        arr = self._data
        if arr.ndim == 3:
            if arr.shape[0] <= 4 and arr.shape[0] < min(arr.shape[1:]):
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=-1)
        return np.asarray(arr, dtype=np.float64)

    def get_roi(self, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        if self._data.ndim == 3:
            if self._data.shape[0] <= 4 and self._data.shape[0] < min(self._data.shape[1:]):
                subset = self._data[:, y0:y1, x0:x1].mean(axis=0)
            else:
                subset = self._data[y0:y1, x0:x1, :].mean(axis=-1)
        else:
            subset = self._data[y0:y1, x0:x1]
        return np.asarray(subset, dtype=np.float64)

    def close(self) -> None:
        try:
            if hasattr(self, "_hdul"):
                self._hdul.close()
        finally:
            try:
                os.remove(self._path)
            except Exception:
                pass


def _open_image_source(data: bytes, filename: Optional[str], probe: Optional[ImageProbe]) -> _BaseImageSource:
    ext = os.path.splitext(filename or "")[1].lower()
    is_fits = (probe is not None and probe.format == "fits") or ext in FIT_EXTENSIONS
    if is_fits and _HAS_ASTROPY:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return _FitsImageSource(tmp.name)
    arr = _to_float_array(data, filename=filename, probe=probe)
    return _ArrayImageSource(arr)


def _compute_roi_bbox(
    positions: Sequence[Tuple[float, float]],
    shape: Tuple[int, int],
    margin: float,
) -> Tuple[int, int, int, int]:
    height, width = shape
    if not positions:
        return 0, width, 0, height
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    x_min = max(min(xs) - margin, 0.0)
    x_max = min(max(xs) + margin, width)
    y_min = max(min(ys) - margin, 0.0)
    y_max = min(max(ys) + margin, height)
    x0 = max(int(math.floor(x_min)), 0)
    x1 = min(int(math.ceil(x_max)), width)
    y0 = max(int(math.floor(y_min)), 0)
    y1 = min(int(math.ceil(y_max)), height)
    if x1 <= x0:
        x1 = min(width, x0 + int(ROI_MIN_MARGIN))
    if y1 <= y0:
        y1 = min(height, y0 + int(ROI_MIN_MARGIN))
    return x0, x1, y0, y1


def _extract_roi_array(
    source: _BaseImageSource,
    positions: Sequence[Tuple[float, float]],
    margin: float,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    bbox = _compute_roi_bbox(positions, source.shape, margin)
    x0, x1, y0, y1 = bbox
    roi = source.get_roi(x0, x1, y0, y1)
    return roi, (x0, y0), bbox

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
    probe: Optional[ImageProbe] = None,
) -> tuple[Image.Image, np.ndarray]:
    """Декодирует изображение и строит превью с растяжкой."""
    arr: np.ndarray
    if (
        probe is not None
        and probe.format == "fits"
        and _HAS_ASTROPY
        and fits is not None
    ) or (
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
        arr = _to_float_array(data, filename=filename, probe=probe)

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
    slot_width = max(1, width // len(chosen))
    for idx, name in enumerate(chosen):
        chart = chart_images[name]
        target_w = max(1, slot_width - 16)
        target_h = max(1, panel_height - 20)
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


def _extract_tile(
    image: Image.Image,
    center: Tuple[float, float],
    radius: float,
    padding: int = 20,
    label: str | None = None,
) -> Image.Image:
    half_size = int(max(radius, 1.0)) + padding
    x, y = center
    left = int(math.floor(x)) - half_size
    top = int(math.floor(y)) - half_size
    right = int(math.floor(x)) + half_size
    bottom = int(math.floor(y)) + half_size

    tile = Image.new("RGB", (right - left, bottom - top), BACKGROUND_COLOR)
    region = image.crop((max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height)))
    paste_x = max(0, -left)
    paste_y = max(0, -top)
    tile.paste(region, (paste_x, paste_y))

    if label:
        draw = ImageDraw.Draw(tile)
        draw.rectangle([4, 4, 4 + 46, 22], fill=(0, 0, 0, 160))
        draw.text((8, 6), label, fill=(255, 255, 255))

    return tile


def _compose_tiles(
    tiles: Sequence[Image.Image],
    layout: str,
    per_row: int,
    padding: int = 16,
) -> Image.Image:
    if not tiles:
        return _placeholder_plot(text="No tiles")

    tile_w, tile_h = tiles[0].size
    if layout == "row":
        cols = len(tiles)
        rows = 1
    else:
        cols = max(1, per_row)
        rows = math.ceil(len(tiles) / cols)

    out_w = cols * tile_w + (cols + 1) * padding
    out_h = rows * tile_h + (rows + 1) * padding
    canvas = Image.new("RGB", (out_w, out_h), BACKGROUND_COLOR)

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + padding)
        canvas.paste(tile, (x, y))

    return canvas


def _measure_morphology(
    image: np.ndarray,
    positions: Sequence[Tuple[float, float]],
    half_size: int = 8,
) -> List[Dict[str, Optional[float]]]:
    if image.ndim == 3:
        image = image.mean(axis=2)

    h, w = image.shape
    results: List[Dict[str, Optional[float]]] = []
    for x, y in positions:
        x0 = max(int(math.floor(x)) - half_size, 0)
        y0 = max(int(math.floor(y)) - half_size, 0)
        x1 = min(int(math.floor(x)) + half_size + 1, w)
        y1 = min(int(math.floor(y)) + half_size + 1, h)
        if x0 >= x1 or y0 >= y1:
            results.append({"fwhm": None, "ellipticity": None, "position_angle": None})
            continue

        cut = image[y0:y1, x0:x1].astype(np.float64, copy=False)
        if cut.size == 0:
            results.append({"fwhm": None, "ellipticity": None, "position_angle": None})
            continue

        background = np.median(cut)
        weights = cut - background
        weights[weights < 0] = 0.0
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0:
            results.append({"fwhm": None, "ellipticity": None, "position_angle": None})
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        xx = xx.astype(np.float64) - x
        yy = yy.astype(np.float64) - y

        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)

        dx = xx - cx
        dy = yy - cy
        cov_xx = float((weights * dx * dx).sum() / total)
        cov_yy = float((weights * dy * dy).sum() / total)
        cov_xy = float((weights * dx * dy).sum() / total)

        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            results.append({"fwhm": None, "ellipticity": None, "position_angle": None})
            continue

        evals = np.clip(evals, a_min=0.0, a_max=None)
        idx_major = int(np.argmax(evals))
        idx_minor = 1 - idx_major
        sigma_major = math.sqrt(float(evals[idx_major])) if evals[idx_major] > 0 else 0.0
        sigma_minor = math.sqrt(float(evals[idx_minor])) if evals[idx_minor] > 0 else 0.0

        if sigma_major <= 0.0:
            fwhm_val = None
            ellipticity = None
        else:
            fwhm_val = FWHM_FACTOR * sigma_major
            ellipticity = 1.0 - (sigma_minor / sigma_major if sigma_major > 0 else 0.0)

        vec = evecs[:, idx_major]
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        if angle < 0:
            angle += 180.0

        results.append(
            {
                "fwhm": float(fwhm_val) if fwhm_val is not None and math.isfinite(fwhm_val) else None,
                "ellipticity": float(ellipticity) if ellipticity is not None and math.isfinite(ellipticity) else None,
                "position_angle": float(angle) if math.isfinite(angle) else None,
            }
        )

    return results


def _apply_global_background(
    results: Sequence[Any],
    positions: Sequence[Tuple[float, float]],
    radius: float,
    bkg_info: Optional[Dict[str, Any]],
    default_mode: str,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> str:
    if not isinstance(results, Sequence) or radius <= 0:
        return default_mode

    if not bkg_info:
        for res in results:
            if isinstance(res, dict):
                res.setdefault("bkg_mode", default_mode)
        return default_mode

    back_map = bkg_info.get("map")
    rms_map = bkg_info.get("rms")
    if back_map is None or rms_map is None:
        for res in results:
            if isinstance(res, dict):
                res.setdefault("bkg_mode", default_mode)
        return default_mode

    back_map = np.asarray(back_map)
    rms_map = np.asarray(rms_map)
    h, w = back_map.shape
    ap_area = math.pi * (float(radius) ** 2)

    for idx, (x, y) in enumerate(positions):
        if idx >= len(results):
            break
        res = results[idx]
        if not isinstance(res, dict):
            continue
        xi = int(round(x - offset[0]))
        yi = int(round(y - offset[1]))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            res.setdefault("bkg_mode", default_mode)
            continue
        bmean = float(back_map[yi, xi])
        brms = float(rms_map[yi, xi]) if rms_map.size else None
        res["bkg_mode"] = "global"
        res["bkg_box"] = bkg_info.get("box")
        res["bkg_filter"] = bkg_info.get("filter")
        res["bkg_clip_sigma"] = bkg_info.get("clip")
        res["bkg_mean"] = bmean
        if brms is not None and math.isfinite(brms):
            res["bkg_rms"] = brms
        ap_sum = res.get("aperture_sum")
        if ap_sum is not None:
            try:
                ap_sum_f = float(ap_sum)
                res["flux_sub"] = ap_sum_f - bmean * ap_area
            except Exception:
                pass
    return "global"

# -----------------------------------------------------------------------------
# Эндпоинты
# -----------------------------------------------------------------------------
@router.get("/health")
def health() -> Dict[str, str]:
    endpoint = _versioned("/health")
    t = _track(endpoint, "GET")
    try:
        payload = {"status": "ok"}
        t.ok()
        return payload
    except Exception as exc:
        t.fail(500)
        raise _service_error("health check failed", hint=str(exc))

# Backwards compatibility: expose unversioned /health
app.add_api_route("/health", health, methods=["GET"])


@router.get("/ready")
async def ready():
    endpoint = _versioned("/ready")
    t = _track(endpoint, "GET")
    checks: Dict[str, Dict[str, Any]] = {}
    status_code = 200

    def _record(name: str, ok: bool, **details: Any) -> None:
        nonlocal status_code
        entry: Dict[str, Any] = {"ok": bool(ok)}
        for key, value in details.items():
            if value is not None:
                entry[key] = value
        if not ok:
            status_code = max(status_code, 503)
        checks[name] = entry

    try:
        _record("sep", sep is not None, error=None if sep is not None else "sep module not available")

        phot_ok = has_real_photometry()
        _record(
            "photometry",
            phot_ok,
            error=None if phot_ok else "astropy/photutils not available",
        )

        tmp_ok = True
        tmp_error = None
        try:
            with tempfile.NamedTemporaryFile(prefix="astro_ready_", delete=True) as fh:
                fh.write(b"ok")
        except Exception as exc:
            tmp_ok = False
            tmp_error = str(exc)
        _record("tmp_write", tmp_ok, error=tmp_error)

        model_info: Dict[str, Any] = dict(_MODEL_INFO)
        model_error: Optional[str] = None
        model_ok = False
        if _HAS_TORCH:
            try:
                start = time.perf_counter()
                _ensure_model()
                model_ok = True
                model_info.setdefault("device", str(_device))
                model_info.setdefault("categories", len(_idx_to_label or []))
                model_info["load_seconds"] = round(time.perf_counter() - start, 4)
                try:  # optional version hints
                    import torchvision  # type: ignore

                    model_info.setdefault("torchvision", getattr(torchvision, "__version__", None))
                except Exception:
                    model_info.setdefault("torchvision", None)
                model_info.setdefault("torch", getattr(torch, "__version__", None) if torch is not None else None)
            except HTTPException as exc:
                detail = exc.detail
                if isinstance(detail, dict):
                    model_error = detail.get("message") or str(detail)
                else:
                    model_error = str(detail)
            except Exception as exc:
                model_error = str(exc)
        else:
            model_error = "torch not installed"

        _record("model", model_ok, error=model_error, **model_info)

        status_text = "ok" if status_code == 200 else "degraded"
        payload = {"status": status_text, "checks": checks}
        if status_code == 200:
            t.ok()
        else:
            t.fail(status_code)
        return JSONResponse(status_code=status_code, content=payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ready check failed")
        t.fail(500)
        raise _service_error("ready check failed", hint=str(exc))


# Backwards compatibility: expose unversioned /ready
app.add_api_route("/ready", ready, methods=["GET"])

@router.get("/metrics")
def metrics():
    """Выдаёт текущие метрики Prometheus из глобального реестра."""
    data = generate_latest(PROM_REGISTRY)  # используем глобальный реестр, не создаём новый
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

# Backwards compatibility: expose unversioned /metrics
app.add_api_route("/metrics", metrics, methods=["GET"])

@router.post("/classify")
async def classify(
    file: UploadFile = File(...),
    topk: int = Query(5, ge=1, le=1000),
    imagenet_norm: bool = Query(True),
):
    endpoint = _versioned("/classify")
    t = _track(endpoint, "POST")
    try:
        _require_torch()
        data = await file.read()
        _validate_upload_size(data)
        if topk < 1:
            raise _validation_error("Parameter topk must be >= 1", hint="Increase topk to 1 or higher.", code="ASTRO_4004")
        with _infer_timer(endpoint):
            results = _classify_bytes(data, imagenet_norm=imagenet_norm, topk=topk, filename=file.filename)
        t.ok()
        return {"filename": file.filename, "results": results}
    except ImageValidationError as exc:
        t.fail(400)
        raise _validation_error(str(exc), hint="Upload a valid image file for classification.", code="ASTRO_4007") from exc
    except Exception as e:
        logger.exception("classify failed")
        t.fail(500)
        raise _service_error("Classification failed", hint=str(e))

@router.post("/classify_batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    topk: int = Query(5, ge=1, le=1000),
    imagenet_norm: bool = Query(True),
):
    endpoint = _versioned("/classify_batch")
    t = _track(endpoint, "POST")
    if topk < 1:
        t.fail(400)
        raise _validation_error("Parameter topk must be >= 1", hint="Increase topk to 1 or higher.", code="ASTRO_4004")
    out = []
    status = 200
    try:
        _require_torch()
    except HTTPException as exc:
        t.fail(exc.status_code)
        raise
    for f in files:
        try:
            data = await f.read()
            _validate_upload_size(data)
            with _infer_timer(endpoint):
                results = _classify_bytes(data, imagenet_norm, topk, filename=f.filename)
            out.append({"filename": f.filename, "results": results})
        except ImageValidationError as exc:
            logger.warning("batch item failed validation: %s", exc)
            status = 207
            out.append({"filename": f.filename, "error": str(exc)})
        except Exception as e:
            logger.exception("batch item failed")
            status = 207  # частично успешно
            out.append({"filename": f.filename, "error": str(e)})

    t.ok(status)
    return {"count": len(files), "results": out}

@router.post("/detect_sources")
async def detect_sources(
    request: Request,
    file: UploadFile = File(..., description="Image file: JPG/PNG/TIFF/FITS*"),
    xy: List[str] = Query(
        default=[],
        description='Repeatable center coords "x,y". Example: &xy=120.5,80.2&xy=30,40',
    ),
    r: float | None = Query(default=None, description="Aperture radius (px)"),
    r_in: float | None = Query(default=None, description="Annulus inner radius (px)"),
    r_out: float | None = Query(default=None, description="Annulus outer radius (px)"),
    r_mode: str = Query("manual", pattern="^(manual|auto)$", description="Aperture selection mode"),
    r_factor: float = Query(2.0, ge=0.5, le=5.0, description="Scale factor for auto aperture (r = factor * FWHM)"),
    apcorr_factor: float = Query(1.5, ge=1.0, le=5.0, description="Multiplier for aperture correction radius"),
    bkg: str = Query("local", pattern="^(local|global)$", description="Background estimation mode"),
    bkg_box: int = Query(32, ge=8, le=512, description="Background box size for global estimation"),
    bkg_filter: int = Query(3, ge=1, le=15, description="Filter size for background smoothing"),
    bkg_clip_sigma: float = Query(3.0, ge=0.5, le=10.0, description="Sigma clipping for background"),
    format: str = Query("json", pattern="^(json|csv|fits)$", description="Export format for photometry results"),
    download: bool = Query(False, description="Force attachment Content-Disposition when true"),
    bundle: str = Query("none", pattern="^(none|zip)$", description="Bundle response into ZIP archive"),
    csv_delimiter: str = Query(",", description="CSV delimiter (use '\\t' or 'tab' for tab)"),
    csv_float_fmt: str = Query(".6f", description="Python format spec for CSV floats"),
    json_indent: int = Query(2, ge=0, le=8, description="Indent for JSON export (ignored when compact=true)"),
    json_compact: bool = Query(False, description="Compact JSON export (overrides indent)"),
    phot_mode: str = Query("aperture", pattern="^(aperture|psf)$", description="Photometry mode"),
    psf_stamp: int = Query(8, ge=4, le=20, description="PSF stamp radius (px)"),
    psf_fit: int = Query(4, ge=2, le=12, description="PSF fit radius (px)"),
    exptime: float | None = Query(None, ge=0.0, description="Exposure time in seconds"),
    gain: float | None = Query(None, ge=0.0, description="Detector gain (e-/ADU)"),
    zeropoint: float | None = Query(None, description="Photometric zero point"),
    mag_system: str = Query("AB", description="Magnitude system label (e.g. AB or Vega)"),
):
    """
    Измерение яркости источников.
    - Если заданы `xy` и `r` и доступны astropy+photutils → апертурная фотометрия.
    - Иначе → быстрая оценка яркости (simple_brightness ~ [0..1]).
    Поддерживает экспорт таблицы: json|csv|fits (+ ZIP bundle).
    """
    endpoint = _versioned("/detect_sources")
    t = _track(endpoint, "POST")
    phot_mode_norm = phot_mode.lower()
    calibration = _CalibrationParams(
        exptime=float(exptime) if exptime is not None else None,
        gain=float(gain) if gain is not None else None,
        zeropoint=float(zeropoint) if zeropoint is not None else None,
        mag_system=mag_system.strip() if mag_system else None,
    )
    try:
        data = await file.read()
        _validate_upload_size(data)
        probe = _preflight_image_bytes(data, file.filename, context="/preview_apertures")
        probe = _preflight_image_bytes(data, file.filename, context="/detect_sources")

        _validate_aperture_triplet(
            r,
            r_in,
            r_out,
            context="/detect_sources",
            allow_missing_r=r_mode == "auto",
        )

        aperture_meta: Dict[str, Any] = {
            "mode": "manual",
            "radius": float(r) if r is not None else None,
            "r": r,
            "r_in": r_in,
            "r_out": r_out,
            "r_factor": r_factor if r_mode == "auto" else None,
            "apcorr_factor": apcorr_factor if r_mode == "auto" else None,
        }

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

        image_source: Optional[_BaseImageSource]
        try:
            image_source = _open_image_source(data, file.filename, probe)
        except Exception:
            image_source = _ArrayImageSource(
                _to_float_array(data, filename=file.filename, probe=probe)
            )

        _validate_pixel_limit(image_source.get_full())

        background_meta: Dict[str, Any] = {"requested": bkg}
        roi_array: Optional[np.ndarray] = None
        roi_offset = (0, 0)
        roi_bbox = (0, image_source.shape[1], 0, image_source.shape[0])

        base_margin = ROI_MIN_MARGIN + 16.0
        if positions:
            base_margin = max(
                ROI_MIN_MARGIN,
                float(r or 0.0),
                float(r_in or 0.0),
                float(r_out or 0.0),
                float((r or 0.0) * apcorr_factor if r else 0.0),
            ) + 16.0
            roi_timer_start = time.perf_counter()
            roi_array, roi_offset, roi_bbox = _extract_roi_array(image_source, positions, base_margin)
            ROI_SECONDS.labels(endpoint).observe(time.perf_counter() - roi_timer_start)
        else:
            roi_array = image_source.get_full()

        background_meta["offset"] = {"x": roi_offset[0], "y": roi_offset[1]}
        background_meta["bbox"] = {
            "x0": roi_bbox[0],
            "x1": roi_bbox[1],
            "y0": roi_bbox[2],
            "y1": roi_bbox[3],
        }

        background_mode_effective = "local"
        global_bkg_info: Optional[Dict[str, Any]] = None

        do_aperture = bool(
            positions
            and has_real_photometry()
            and ((r is not None) or r_mode == "auto")
        )

        if do_aperture:
            roi_arr = roi_array if roi_array is not None else image_source.get_full()
            positions_roi = [(px - roi_offset[0], py - roi_offset[1]) for px, py in positions]

            morph_metrics: List[Dict[str, Optional[float]]]
            try:
                morph_metrics = _measure_morphology(roi_arr, positions_roi)
            except Exception:
                morph_metrics = []

            auto_radius: Optional[float] = r
            auto_used = False
            if r_mode == "auto":
                fwhm_vals = [m.get("fwhm") for m in morph_metrics if m.get("fwhm")]
                if fwhm_vals:
                    auto_radius = max(0.5, float(np.median(fwhm_vals)) * r_factor)
                    auto_used = True
                elif r is not None:
                    auto_radius = r
                else:
                    raise _validation_error(
                        "Auto aperture failed: could not estimate FWHM",
                        hint="Provide r manually or ensure sources are detectable.",
                    )

            radius_to_use = float(auto_radius if auto_radius is not None else r or 0)
            if radius_to_use <= 0:
                raise _validation_error("Aperture radius must be > 0")

            required_margin = max(
                ROI_MIN_MARGIN,
                radius_to_use,
                float(r_in or 0.0),
                float(r_out or 0.0),
                radius_to_use * apcorr_factor,
            ) + 16.0

            if positions and required_margin > base_margin + 1.0:
                roi_timer_start = time.perf_counter()
                roi_arr, roi_offset, roi_bbox = _extract_roi_array(image_source, positions, required_margin)
                ROI_SECONDS.labels(endpoint).observe(time.perf_counter() - roi_timer_start)
                positions_roi = [(px - roi_offset[0], py - roi_offset[1]) for px, py in positions]
                background_meta["offset"] = {"x": roi_offset[0], "y": roi_offset[1]}
                background_meta["bbox"] = {
                    "x0": roi_bbox[0],
                    "x1": roi_bbox[1],
                    "y0": roi_bbox[2],
                    "y1": roi_bbox[3],
                }
                base_margin = required_margin
                try:
                    morph_metrics = _measure_morphology(roi_arr, positions_roi)
                except Exception:
                    morph_metrics = []

            def _compute_background(array: np.ndarray) -> Tuple[str, Optional[Dict[str, Any]]]:
                if bkg != "global" or sep is None:
                    return ("local" if bkg == "local" else "local-fallback", None)
                start = time.perf_counter()
                bw = max(8, int(bkg_box))
                bh = bw
                fw = max(1, int(bkg_filter))
                if fw % 2 == 0:
                    fw += 1
                try:
                    background = sep.Background(
                        array.astype(np.float32, copy=False),
                        bw=bw,
                        bh=bh,
                        fw=fw,
                        fh=fw,
                    )
                    back_map = background.back()
                    rms_map = background.rms()
                    if bkg_clip_sigma > 0:
                        med = float(np.median(back_map))
                        std = float(np.std(back_map))
                        if std > 0:
                            limit = bkg_clip_sigma * std
                            back_map = np.clip(back_map, med - limit, med + limit)
                    info = {
                        "map": back_map,
                        "rms": rms_map,
                        "box": bw,
                        "filter": fw,
                        "clip": bkg_clip_sigma,
                        "offset": roi_offset,
                    }
                    return "global", info
                except Exception:
                    return "local-fallback", None
                finally:
                    BACKGROUND_SECONDS.labels(endpoint).observe(time.perf_counter() - start)

            background_mode_effective, global_bkg_info = _compute_background(roi_arr)
            background_meta["mode"] = background_mode_effective

            with _infer_timer(endpoint):
                with TRACER.start_as_current_span("photometry.measure_brightness") as span:
                    if span is not None:
                        span.set_attribute("astro.detector", "manual")
                        span.set_attribute("astro.positions", len(positions_roi))
                        span.set_attribute("astro.aperture_r", float(radius_to_use))
                    results = measure_brightness(
                        roi_arr,
                        positions=positions_roi,
                        r=radius_to_use,
                        r_in=r_in,
                        r_out=r_out,
                    )

            background_mode_effective = _apply_global_background(
                results,
                positions,
                radius_to_use,
                global_bkg_info,
                background_mode_effective,
                offset=roi_offset,
            )
            background_meta["mode"] = background_mode_effective

            for idx, morph in enumerate(morph_metrics):
                if idx < len(results) and isinstance(results[idx], dict):
                    for key, value in morph.items():
                        if value is not None:
                            results[idx][key] = value

            if auto_used and apcorr_factor > 1.0:
                ap_radius = radius_to_use * apcorr_factor
                use_roi_for_apcorr = ap_radius <= required_margin
                try:
                    if use_roi_for_apcorr:
                        with TRACER.start_as_current_span("photometry.measure_brightness") as span:
                            if span is not None:
                                span.set_attribute("astro.detector", "manual")
                                span.set_attribute("astro.positions", len(positions_roi))
                                span.set_attribute("astro.aperture_r", float(ap_radius))
                            big_results = measure_brightness(
                                roi_arr,
                                positions=positions_roi,
                                r=ap_radius,
                                r_in=r_in,
                                r_out=r_out,
                            )
                    else:
                        with TRACER.start_as_current_span("photometry.measure_brightness") as span:
                            if span is not None:
                                span.set_attribute("astro.detector", "manual")
                                span.set_attribute("astro.positions", len(positions))
                                span.set_attribute("astro.aperture_r", float(ap_radius))
                            big_results = measure_brightness(
                                image_source.get_full(),
                                positions=positions,
                                r=ap_radius,
                                r_in=r_in,
                                r_out=r_out,
                            )
                    for idx, res in enumerate(results):
                        if idx < len(big_results) and isinstance(res, dict) and isinstance(big_results[idx], dict):
                            small_flux = res.get("flux_sub") or res.get("aperture_sum")
                            large_flux = big_results[idx].get("flux_sub") or big_results[idx].get("aperture_sum")
                            if small_flux and large_flux and small_flux != 0:
                                res["apcorr"] = float(large_flux) / float(small_flux)
                except Exception:
                    pass

            for idx, res in enumerate(results):
                if isinstance(res, dict):
                    if idx < len(positions_roi):
                        res["x"] = float(res.get("x", positions_roi[idx][0])) + roi_offset[0]
                        res["y"] = float(res.get("y", positions_roi[idx][1])) + roi_offset[1]

            if auto_used:
                for idx, res in enumerate(results):
                    if isinstance(res, dict):
                        res.setdefault("aperture_mode", "auto")
                        res.setdefault("aperture_radius", radius_to_use)
                aperture_meta.update(
                    {
                        "mode": "auto",
                        "radius": radius_to_use,
                    }
                )
            else:
                for idx, res in enumerate(results):
                    if isinstance(res, dict):
                        res.setdefault("aperture_mode", "manual")
                        if r is not None:
                            res.setdefault("aperture_radius", float(r))
                if r is not None:
                    aperture_meta["radius"] = float(r)

            background_meta.update(
                {
                    "mode": background_mode_effective,
                    "box": global_bkg_info.get("box") if background_mode_effective == "global" and global_bkg_info else None,
                    "filter": global_bkg_info.get("filter") if background_mode_effective == "global" and global_bkg_info else None,
                    "clip_sigma": global_bkg_info.get("clip") if background_mode_effective == "global" and global_bkg_info else None,
                }
            )

            # метрики
            counter_mode = "psf" if phot_mode_norm == "psf" else "aperture"
            PHOT_COUNTER.labels(mode=counter_mode).inc()
            SOURCES_COUNTER.labels(detector="manual").inc(len(positions))

            extras: Dict[str, Any] = {}
            if isinstance(results, list) and results:
                extras = _finalise_photometry_results(
                    image_source.get_full(),
                    results,
                    positions,
                    phot_mode=phot_mode_norm,
                    aperture_radius=r,
                    psf_stamp_radius=psf_stamp,
                    psf_fit_radius=psf_fit,
                    calibration=calibration,
                )
            elif calibration.zeropoint is not None:
                extras["calibration"] = {
                    "zeropoint": calibration.zeropoint,
                    "mag_system": (calibration.mag_system or "AB").lower(),
                    **({"exptime": calibration.exptime} if calibration.exptime is not None else {}),
                    **({"gain": calibration.gain} if calibration.gain is not None else {}),
                }
            extras["photometry_mode"] = phot_mode_norm

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
                    "aperture": aperture_meta,
                    "background": background_meta,
                    "photometry_mode": phot_mode_norm,
                }
                payload.update(extras)
                t.ok()
                return payload

            try:
                with TRACER.start_as_current_span("photometry.export") as span:
                    if span is not None:
                        span.set_attribute("astro.export.format", format_mode)
                        span.set_attribute("astro.export.count", len(results))
                        span.set_attribute("astro.export.detector", "manual")
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

            json_export_payload: Optional[bytes] = None
            preview_bytes: Optional[bytes] = None
            dpi_tuple = (96, 96)

            filename_root = os.path.splitext(file.filename or "image")[0]
            disposition = "attachment" if download else "inline"

            if bundle_mode == "zip":
                json_payload = {
                    "filename": file.filename,
                    "mode": "aperture",
                    "count": len(positions),
                    "aperture": aperture_meta,
                    "background": background_meta,
                    "results": results,
                }
                json_payload.update(extras)
                json_export_payload = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")

                preview_buf = io.BytesIO()
                try:
                    with TRACER.start_as_current_span("preview.bundle_render") as span:
                        if span is not None:
                            span.set_attribute("astro.positions", len(positions))
                            span.set_attribute("astro.preview.layout", "bundle")
                            span.set_attribute("astro.detector", "manual")
                        base_img = Image.open(io.BytesIO(data)).convert("RGB")
                        overlay_preview = _draw_circles(
                            base_img,
                            positions,
                            radius_to_use,
                            r_in,
                            r_out,
                            line=2,
                        )
                        overlay_preview = _add_labels(
                            overlay_preview,
                            positions,
                            {
                                "plots": [],
                                "layout": "bundle",
                                "count_positions": len(positions),
                            },
                        )
                        overlay_to_save = overlay_preview.convert("RGBA") if save_alpha else overlay_preview
                        overlay_to_save.save(preview_buf, format="PNG", dpi=dpi_tuple)
                        preview_bytes = preview_buf.getvalue()
                except Exception:
                    preview_bytes = None

            metadata = {
                "endpoint": _versioned("/detect_sources"),
                "filename": file.filename,
                "aperture": aperture_meta,
                "background": background_meta,
                "count": len(results),
                "format": artifact.extension,
                "sensor": _sensor_label(
                    request,
                    file.filename,
                    image_source.shape if image_source is not None else None,
                ),
                "photometry_mode": phot_mode_norm,
            }
            for key in ("psf_model", "calibration"):
                if key in extras:
                    metadata[key] = extras[key]

            if bundle_mode == "zip":
                with TRACER.start_as_current_span("photometry.bundle") as span:
                    if span is not None:
                        span.set_attribute("astro.bundle.format", "zip")
                        span.set_attribute("astro.bundle.count", len(results))
                        span.set_attribute("astro.export.detector", "manual")
                    zip_bytes = build_zip_bundle(
                        artifact,
                        metadata=metadata,
                        filename_stem=f"{filename_root}.photometry",
                        preview_png=preview_bytes,
                        json_payload=json_export_payload,
                    )
                EXPORT_BYTES_COUNTER.labels(format="zip").inc(len(zip_bytes))
                headers = {
                    "Content-Disposition": f'{disposition}; filename="{filename_root}.photometry_bundle.zip"',
                    "X-Astro-Columns": ",".join(artifact.columns),
                }
                t.ok()
                return Response(content=zip_bytes, media_type="application/zip", headers=headers)

            filename_export = f"{filename_root}.photometry.{artifact.extension}"
            fmt_label = _sanitize_label(artifact.extension, "unknown")
            EXPORT_BYTES_COUNTER.labels(format=fmt_label).inc(len(artifact.content))
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

        background_meta.update(
            {
                "mode": "simple",
                "box": None,
                "filter": None,
                "clip_sigma": None,
            }
        )

        background_meta.setdefault("mode", "simple")
        background_meta.setdefault("offset", {"x": 0, "y": 0})
        background_meta.setdefault("bbox", {
            "x0": 0,
            "x1": image_source.shape[1] if image_source else 0,
            "y0": 0,
            "y1": image_source.shape[0] if image_source else 0,
        })
        val = simple_brightness(data)
        payload = {
            "filename": file.filename,
            "mode": "simple",
            "real_photometry": has_real_photometry(),
            "simple_brightness": val,
            "background": background_meta,
        }
        payload["photometry_mode"] = "simple"
        if calibration.zeropoint is not None:
            payload["calibration"] = {
                "zeropoint": calibration.zeropoint,
                "mag_system": (calibration.mag_system or "AB").lower(),
                **({"exptime": calibration.exptime} if calibration.exptime is not None else {}),
                **({"gain": calibration.gain} if calibration.gain is not None else {}),
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
    finally:
        if image_source is not None:
            image_source.close()

@router.post("/detect_auto")
async def detect_auto(
    request: Request,
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
    r_mode: str = Query("manual", pattern="^(manual|auto)$", description="Aperture selection mode"),
    r_factor: float = Query(2.0, ge=0.5, le=5.0, description="Scale factor for auto aperture (r = factor * FWHM)"),
    apcorr_factor: float = Query(1.5, ge=1.0, le=5.0, description="Multiplier for aperture correction"),
    bkg: str = Query("local", pattern="^(local|global)$", description="Background estimation mode"),
    bkg_box: int = Query(32, ge=8, le=512, description="Background box size for global estimation"),
    bkg_filter: int = Query(3, ge=1, le=15, description="Filter size for background smoothing"),
    bkg_clip_sigma: float = Query(3.0, ge=0.5, le=10.0, description="Sigma clipping for background"),
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
    phot_mode: str = Query("aperture", pattern="^(aperture|psf)$", description="Photometry mode"),
    psf_stamp: int = Query(8, ge=4, le=20, description="PSF stamp radius (px)"),
    psf_fit: int = Query(4, ge=2, le=12, description="PSF fit radius (px)"),
    exptime: float | None = Query(None, ge=0.0, description="Exposure time in seconds"),
    gain: float | None = Query(None, ge=0.0, description="Detector gain (e-/ADU)"),
    zeropoint: float | None = Query(None, description="Photometric zero point"),
    mag_system: str = Query("AB", description="Magnitude system label (e.g. AB or Vega)"),
):
    """
    Автопоиск источников (DAOStarFinder или SEP) + апертурная фотометрия по найденным центрам.
    Поддерживает экспорт результатов фотометрии: json|csv|fits (+ ZIP bundle).
    """
    endpoint = _versioned("/detect_auto")
    t = _track(endpoint, "POST")
    phot_mode_norm = phot_mode.lower()
    calibration = _CalibrationParams(
        exptime=float(exptime) if exptime is not None else None,
        gain=float(gain) if gain is not None else None,
        zeropoint=float(zeropoint) if zeropoint is not None else None,
        mag_system=mag_system.strip() if mag_system else None,
    )

    if not has_real_photometry():
        t.fail(502)
        raise _dependency_error("Real photometry (photutils/astropy) is required.")

    if detector == "dao" and DAOStarFinder is None:
        t.fail(502)
        raise _dependency_error("DAOStarFinder requires photutils.detection")

    _validate_aperture_triplet(
        r,
        r_in,
        r_out,
        context="/detect_auto",
        allow_missing_r=r_mode == "auto",
    )

    background_mode_effective = "local"
    global_bkg_info: Optional[Dict[str, Any]] = None
    background_meta: Dict[str, Any] = {"requested": bkg}

    try:
        data = await file.read()
        _validate_upload_size(data)
        probe = _preflight_image_bytes(data, file.filename, context="/detect_auto")

        # Подготавливаем изображение (градации серого + нормализация)
        arr = _to_float_array(data, filename=file.filename, probe=probe)
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
        auto_used = False
        radius_to_use = float(r)

        if detector == "sep":
            if sep is None:
                t.fail(502)
                raise _dependency_error("SEP is not installed; pip install sep")

            data32 = np.ascontiguousarray(arr.astype(np.float32))
            bkg_sep = sep.Background(data32)
            data_sub = data32 - bkg_sep.back()
            thresh_abs = threshold_sigma * bkg_sep.globalrms
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

        if bkg == "global":
            start = time.perf_counter()
            if sep is None:
                background_mode_effective = "local-fallback"
            else:
                bw = max(8, int(bkg_box))
                bh = bw
                fw = max(1, int(bkg_filter))
                if fw % 2 == 0:
                    fw += 1
                try:
                    background = sep.Background(
                        arr.astype(np.float32, copy=False),
                        bw=bw,
                        bh=bh,
                        fw=fw,
                        fh=fw,
                    )
                    back_map = background.back()
                    rms_map = background.rms()
                    if bkg_clip_sigma > 0:
                        med_back = float(np.median(back_map))
                        std_back = float(np.std(back_map))
                        if std_back > 0:
                            limit = bkg_clip_sigma * std_back
                            back_map = np.clip(back_map, med_back - limit, med_back + limit)
                    global_bkg_info = {
                        "map": back_map,
                        "rms": rms_map,
                        "box": bw,
                        "filter": fw,
                        "clip": bkg_clip_sigma,
                    }
                    background_mode_effective = "global"
                except Exception:
                    global_bkg_info = None
                    background_mode_effective = "local-fallback"
                finally:
                    BACKGROUND_SECONDS.labels(endpoint).observe(time.perf_counter() - start)

        if not positions:
            background_meta.update(
                {
                    "mode": background_mode_effective,
                    "box": global_bkg_info.get("box") if background_mode_effective == "global" and global_bkg_info else None,
                    "filter": global_bkg_info.get("filter") if background_mode_effective == "global" and global_bkg_info else None,
                    "clip_sigma": global_bkg_info.get("clip") if background_mode_effective == "global" and global_bkg_info else None,
                }
            )

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
                "background": background_meta,
                "meta": {
                    "fwhm": fwhm,
                    "threshold_sigma": threshold_sigma,
                    "threshold_abs": threshold,
                    "robust_sigma": robust_sigma,
                    "aperture_mode": "auto" if auto_used else "manual",
                    "aperture_radius": radius_to_use,
                    "r_factor": r_factor if auto_used else None,
                    "apcorr_factor": apcorr_factor if auto_used else None,
                    "background_mode": background_mode_effective,
                    "sep_minarea": sep_minarea if detector == "sep" else None,
                    "sep_filter_kernel": sep_filter_kernel if detector == "sep" else None,
                    "sep_deblend_nthresh": sep_deblend_nthresh if detector == "sep" else None,
                    "sep_deblend_cont": sep_deblend_cont if detector == "sep" else None,
                    "dao_sharplo": dao_sharplo if detector == "dao" else None,
                    "dao_roundlo": dao_roundlo if detector == "dao" else None,
                    "dao_roundhi": dao_roundhi if detector == "dao" else None,
                },
            }

        morph_metrics = _measure_morphology(arr, positions)

        if r_mode == "auto":
            fwhm_vals = [m.get("fwhm") for m in morph_metrics if m.get("fwhm")]
            if fwhm_vals:
                radius_to_use = max(1.0, float(np.median(fwhm_vals)) * r_factor)
                auto_used = True
            else:
                radius_to_use = float(r)
        else:
            radius_to_use = float(r)

        with _infer_timer(endpoint):
            with TRACER.start_as_current_span("photometry.measure_brightness") as span:
                if span is not None:
                    span.set_attribute("astro.detector", detector)
                    span.set_attribute("astro.positions", len(positions))
                    span.set_attribute("astro.aperture_r", float(radius_to_use))
                phot = measure_brightness(
                    data, positions=positions, r=radius_to_use, r_in=r_in, r_out=r_out
                )

        background_mode_effective = _apply_global_background(
            phot,
            positions,
            radius_to_use,
            global_bkg_info,
            background_mode_effective,
            offset=(0.0, 0.0),
        )

        background_meta.update(
            {
                "mode": background_mode_effective,
                "box": global_bkg_info.get("box") if background_mode_effective == "global" and global_bkg_info else None,
                "filter": global_bkg_info.get("filter") if background_mode_effective == "global" and global_bkg_info else None,
                "clip_sigma": global_bkg_info.get("clip") if background_mode_effective == "global" and global_bkg_info else None,
            }
        )

        aperture_meta = {
            "mode": "auto" if auto_used else "manual",
            "radius": radius_to_use,
            "r_factor": r_factor if auto_used else None,
            "apcorr_factor": apcorr_factor if auto_used else None,
            "r_in": r_in,
            "r_out": r_out,
        }

        for idx, morph in enumerate(morph_metrics):
            if idx < len(phot) and isinstance(phot[idx], dict):
                for key, value in morph.items():
                    if value is not None:
                        phot[idx][key] = value

        if auto_used and apcorr_factor > 1.0:
            try:
                with TRACER.start_as_current_span("photometry.measure_brightness") as span:
                    if span is not None:
                        span.set_attribute("astro.detector", detector)
                        span.set_attribute("astro.positions", len(positions))
                        span.set_attribute("astro.aperture_r", float(radius_to_use * apcorr_factor))
                    big_results = measure_brightness(
                        data, positions=positions, r=radius_to_use * apcorr_factor, r_in=r_in, r_out=r_out
                    )
                _apply_global_background(
                    big_results,
                    positions,
                    radius_to_use * apcorr_factor,
                    global_bkg_info,
                    background_mode_effective,
                    offset=(0.0, 0.0),
                )
                for idx, res in enumerate(phot):
                    if idx < len(big_results) and isinstance(res, dict) and isinstance(big_results[idx], dict):
                        small_flux = res.get("flux_sub")
                        large_flux = big_results[idx].get("flux_sub")
                        if small_flux and large_flux and small_flux != 0:
                            res["apcorr"] = float(large_flux) / float(small_flux)
            except Exception:
                pass

        for res in phot:
            if isinstance(res, dict):
                res.setdefault("aperture_mode", "auto" if auto_used else "manual")
                res.setdefault("aperture_radius", radius_to_use)

        # метрики
        counter_mode = "psf" if phot_mode_norm == "psf" else "aperture"
        PHOT_COUNTER.labels(mode=counter_mode).inc()
        SOURCES_COUNTER.labels(detector=detector).inc(len(positions))

        extras: Dict[str, Any] = {}
        if isinstance(phot, list) and phot:
            extras = _finalise_photometry_results(
                arr,
                phot,
                positions,
                phot_mode=phot_mode_norm,
                aperture_radius=radius_to_use,
                psf_stamp_radius=psf_stamp,
                psf_fit_radius=psf_fit,
                calibration=calibration,
            )
        elif calibration.zeropoint is not None:
            extras["calibration"] = {
                "zeropoint": calibration.zeropoint,
                "mag_system": (calibration.mag_system or "AB").lower(),
                **({"exptime": calibration.exptime} if calibration.exptime is not None else {}),
                **({"gain": calibration.gain} if calibration.gain is not None else {}),
            }
        extras["photometry_mode"] = phot_mode_norm

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
            payload = {
                "filename": file.filename,
                "mode": "auto-aperture",
                "detector": detector,
                "real_photometry": True,
                "count": len(positions),
                "positions": [{"x": x, "y": y} for x, y in positions],
                "results": phot,
                "aperture": aperture_meta,
                "background": background_meta,
                "photometry_mode": phot_mode_norm,
                "meta": {
                    "fwhm": fwhm,
                    "threshold_sigma": threshold_sigma,
                    "threshold_abs": threshold,
                    "robust_sigma": robust_sigma,
                    "background_mode": background_mode_effective,
                    "sep_minarea": sep_minarea if detector == "sep" else None,
                    "sep_filter_kernel": sep_filter_kernel if detector == "sep" else None,
                    "sep_deblend_nthresh": sep_deblend_nthresh if detector == "sep" else None,
                    "sep_deblend_cont": sep_deblend_cont if detector == "sep" else None,
                    "dao_sharplo": dao_sharplo if detector == "dao" else None,
                    "dao_roundlo": dao_roundlo if detector == "dao" else None,
                    "dao_roundhi": dao_roundhi if detector == "dao" else None,
                },
            }
            payload.update(extras)
            t.ok()
            return payload

        try:
            with TRACER.start_as_current_span("photometry.export") as span:
                if span is not None:
                    span.set_attribute("astro.export.format", format_mode)
                    span.set_attribute("astro.export.count", len(phot))
                    span.set_attribute("astro.export.detector", detector)
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

        json_export_payload: Optional[bytes] = None
        preview_bytes: Optional[bytes] = None
        dpi_tuple = (96, 96)

        metadata = {
            "endpoint": _versioned("/detect_auto"),
            "filename": file.filename,
            "detector": detector,
            "aperture": aperture_meta,
            "background": background_meta,
            "count": len(positions),
            "format": artifact.extension,
            "sensor": _sensor_label(request, file.filename, arr.shape),
            "photometry_mode": phot_mode_norm,
            "meta": {
                "fwhm": fwhm,
                "threshold_sigma": threshold_sigma,
                "threshold_abs": threshold,
                "robust_sigma": robust_sigma,
                "aperture_mode": "auto" if auto_used else "manual",
                "aperture_radius": radius_to_use,
                "r_factor": r_factor if auto_used else None,
                "apcorr_factor": apcorr_factor if auto_used else None,
                "background_mode": background_mode_effective,
                "sep_minarea": sep_minarea if detector == "sep" else None,
                "sep_filter_kernel": sep_filter_kernel if detector == "sep" else None,
                "sep_deblend_nthresh": sep_deblend_nthresh if detector == "sep" else None,
                "sep_deblend_cont": sep_deblend_cont if detector == "sep" else None,
                "dao_sharplo": dao_sharplo if detector == "dao" else None,
                "dao_roundlo": dao_roundlo if detector == "dao" else None,
                "dao_roundhi": dao_roundhi if detector == "dao" else None,
            },
        }
        for key in ("psf_model", "calibration"):
            if key in extras:
                metadata[key] = extras[key]

        if bundle_mode == "zip":
            json_payload = {
                "filename": file.filename,
                "mode": "auto-aperture",
                "detector": detector,
                "count": len(positions),
                "positions": [{"x": x, "y": y} for x, y in positions],
                "results": phot,
                "aperture": aperture_meta,
                "background": background_meta,
            }
            json_payload.update(extras)
            json_export_payload = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")
            try:
                with TRACER.start_as_current_span("preview.bundle_render") as span:
                    if span is not None:
                        span.set_attribute("astro.positions", len(positions))
                        span.set_attribute("astro.preview.layout", "bundle")
                        span.set_attribute("astro.detector", detector)
                    base_img = Image.open(io.BytesIO(data)).convert("RGB")
                    overlay_preview = _draw_circles(
                        base_img,
                        positions,
                        radius_to_use,
                        r_in,
                        r_out,
                        line=2,
                    )
                    overlay_preview = _add_labels(
                        overlay_preview,
                        positions,
                        {
                            "plots": [],
                            "layout": "bundle",
                            "count_positions": len(positions),
                        },
                    )
                    overlay_to_save = overlay_preview.convert("RGBA") if save_alpha else overlay_preview
                    preview_buf = io.BytesIO()
                    overlay_to_save.save(preview_buf, format="PNG", dpi=dpi_tuple)
                    preview_bytes = preview_buf.getvalue()
            except Exception:
                preview_bytes = None

        if bundle_mode == "zip":
            with TRACER.start_as_current_span("photometry.bundle") as span:
                if span is not None:
                    span.set_attribute("astro.bundle.format", "zip")
                    span.set_attribute("astro.bundle.count", len(phot))
                    span.set_attribute("astro.export.detector", detector)
                zip_bytes = build_zip_bundle(
                    artifact,
                    metadata=metadata,
                    filename_stem=f"{filename_root}.auto.photometry",
                    preview_png=preview_bytes,
                    json_payload=json_export_payload,
                )
            EXPORT_BYTES_COUNTER.labels(format="zip").inc(len(zip_bytes))
            headers = {
                "Content-Disposition": f'{disposition}; filename="{filename_root}.auto.photometry_bundle.zip"',
                "X-Astro-Columns": ",".join(artifact.columns),
            }
            t.ok()
            return Response(content=zip_bytes, media_type="application/zip", headers=headers)

        fmt_label = _sanitize_label(artifact.extension, "unknown")
        EXPORT_BYTES_COUNTER.labels(format=fmt_label).inc(len(artifact.content))
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


@router.get("/cutout")
async def fetch_cutout(
    service: str = Query(..., description="Data source identifier: noirlab|mast|sdss|ztf"),
    ra: float = Query(..., description="Right ascension in degrees"),
    dec: float = Query(..., description="Declination in degrees"),
    size_deg: float | None = Query(None, ge=0.0, description="Cutout size in degrees"),
    size_arcmin: float | None = Query(None, ge=0.0, description="Cutout size in arcminutes"),
    size_arcsec: float | None = Query(None, ge=0.0, description="Cutout size in arcseconds"),
    band: str | None = Query(None, description="Band/passband filter"),
    filters: List[str] = Query(default=[], description="Filter list for MAST search"),
):
    endpoint = _versioned("/cutout")
    t = _track(endpoint, "GET")
    try:
        provider = get_cutout_provider(service)
    except ValueError as exc:
        t.fail(400)
        raise _validation_error(str(exc))

    request_obj = CutoutRequest(
        ra=float(ra),
        dec=float(dec),
        size_deg=size_deg,
        size_arcmin=size_arcmin,
        size_arcsec=size_arcsec,
        band=band,
        filters=[f for f in filters if f],
    )
    try:
        result = provider.fetch(request_obj)
    except NotImplementedError as exc:
        t.fail(501)
        raise _dependency_error(str(exc))
    except CutoutError as exc:
        t.fail(404)
        raise _astro_error(404, "ASTRO_4041", str(exc), hint="Adjust position or size.")
    except requests.RequestException as exc:
        t.fail(503)
        raise _service_error("Cutout request failed", hint=str(exc))
    except Exception as exc:
        logger.exception("cutout fetch failed")
        t.fail(500)
        raise _service_error("cutout fetch failed", hint=str(exc))

    t.ok()
    headers = {
        "Content-Disposition": f'attachment; filename="{result.filename}"',
        "X-Astro-Provider": provider.name,
    }
    return StreamingResponse(io.BytesIO(result.content), media_type=result.media_type, headers=headers)

@router.post("/preview_apertures")
async def preview_apertures(
    request: Request,
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
    layout: str = Query("overlay", description="Layout: overlay, panel, grid, or row"),
    per_row: int = Query(3, ge=1, le=12, description="Number of tiles per row for grid/row layouts"),
    dpi: float = Query(96.0, ge=30.0, le=600.0, description="Output DPI metadata"),
    save_alpha: bool = Query(False, description="Preserve alpha channel in the resulting PNG"),
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
    endpoint = _versioned("/preview_apertures")
    t = _track(endpoint, "POST")
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
            probe.format == "fits"
            and not _HAS_ASTROPY
        ) or (
            file.filename
            and os.path.splitext(file.filename)[1].lower() in FIT_EXTENSIONS
            and not _HAS_ASTROPY
        ):
            t.fail(502)
            raise _dependency_error("FITS preview requires astropy to be installed")

        png_start = time.perf_counter()
        with TRACER.start_as_current_span("preview.load_image") as span:
            if span is not None:
                span.set_attribute("astro.preview.layout", layout_mode)
                span.set_attribute("astro.positions", len(positions))
                span.set_attribute("astro.request.bundle", bundle_mode)
            preview_img, array_data = _load_preview_arrays(
                data=data,
                filename=file.filename,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                stretch=stretch_mode,
                probe=probe,
            )
        array_shape: Optional[Sequence[int]] = tuple(array_data.shape) if hasattr(array_data, "shape") else None
        _validate_pixel_limit(array_data)
        with TRACER.start_as_current_span("preview.render_overlay") as span:
            if span is not None:
                span.set_attribute("astro.preview.layout", layout_mode)
                span.set_attribute("astro.positions", len(positions))
                span.set_attribute("astro.request.bundle", bundle_mode)
            overlay = _draw_circles(preview_img, positions, r=r, r_in=r_in, r_out=r_out, line=line)

        profile_start = time.perf_counter()
        profiles: List[Dict[str, Any]] = []
        if selected_plots:
            try:
                with TRACER.start_as_current_span("preview.profile_series") as span:
                    if span is not None:
                        span.set_attribute("astro.preview.layout", layout_mode)
                        span.set_attribute("astro.positions", len(positions))
                        span.set_attribute("astro.profile.max_r", float(profile_max_r))
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

        detector_label, format_label, sensor_label, pixels_bin = _preview_metric_labels(
            detector="manual",
            response_format="zip" if bundle_mode == "zip" else "png",
            request=request,
            filename=file.filename,
            array_shape=array_shape,
        )
        PREVIEW_PNG_SECONDS.labels(detector_label, format_label, sensor_label, pixels_bin).observe(png_seconds)
        PREVIEW_PLOTS_SECONDS.labels(detector_label, format_label, sensor_label, pixels_bin).observe(plots_seconds)

        label_payload = {
            "plots": selected_plots,
            "layout": layout_mode,
            "count_positions": len(positions),
        }
        overlay_with_labels = _add_labels(overlay, positions, label_payload) if labels else overlay

        if layout_mode == "panel":
            final_img, plots_img = _compose_panel(overlay_with_labels, chart_images, selected_plots)
        elif layout_mode in {"grid", "row"}:
            tile_source = overlay
            tile_radius = max(
                r_out or 0.0,
                r_in or 0.0,
                r or 0.0,
            )
            tile_radius = tile_radius if tile_radius > 0 else 10.0
            tile_padding = max(12, int(tile_radius * 0.4))
            tiles = []
            for idx, (x, y) in enumerate(positions):
                tiles.append(
                    _extract_tile(
                        tile_source,
                        (x, y),
                        radius=tile_radius,
                        padding=tile_padding,
                        label=f"P{idx + 1}"
                    )
                )
            final_img = _compose_tiles(tiles, layout_mode, per_row)
            plots_img = None
        else:
            final_img, plots_img = _compose_overlay(overlay_with_labels, chart_images, selected_plots)

        filename_root = os.path.splitext(file.filename or "image")[0]
        metrics_payload = {
            "astro_preview_png_seconds": png_seconds,
            "astro_preview_plots_seconds": plots_seconds,
            "labels": label_payload,
        }

        preview_output = final_img

        dpi_value = float(dpi)
        dpi_tuple = (int(round(dpi_value)), int(round(dpi_value)))
        save_kwargs = {"dpi": dpi_tuple}

        if bundle_mode == "zip":
            with TRACER.start_as_current_span("preview.bundle") as span:
                if span is not None:
                    span.set_attribute("astro.preview.layout", layout_mode)
                    span.set_attribute("astro.positions", len(positions))
                    span.set_attribute("astro.request.bundle", "zip")
                    span.set_attribute("astro.plots.count", len(chart_images))
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    preview_buf = io.BytesIO()
                    preview_to_save = preview_output.convert("RGBA") if save_alpha else preview_output
                    preview_to_save.save(preview_buf, format="PNG", **save_kwargs)
                    zf.writestr("preview.png", preview_buf.getvalue())

                    plots_buf = io.BytesIO()
                    if plots_img is not None:
                        plots_save = plots_img.convert("RGBA") if save_alpha else plots_img
                        plots_save.save(plots_buf, format="PNG", **save_kwargs)
                    else:
                        _placeholder_plot().save(plots_buf, format="PNG", **save_kwargs)
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
        final_to_save = preview_output.convert("RGBA") if save_alpha else preview_output
        with TRACER.start_as_current_span("preview.encode") as span:
            if span is not None:
                span.set_attribute("astro.preview.layout", layout_mode)
                span.set_attribute("astro.positions", len(positions))
                span.set_attribute("astro.request.bundle", "png")
            final_to_save.save(buf, format="PNG", **save_kwargs)
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

app.include_router(router)
