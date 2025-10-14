# api/main.py
from __future__ import annotations

import io
import os
import re
import math
import time
import json
import csv
import logging
import threading
import warnings
import urllib.parse
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- FastAPI ---
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
    Response,
)

# --- Images / drawing ---
from PIL import Image, ImageDraw, ImageDecompressionBombError, ImageOps

# --- NumPy ---
import numpy as np

# --- Torch / torchvision ---
import torch
from torch import nn
from torchvision import models, transforms

# --- Prometheus ---
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
)

# --- Optional: FITS / Astropy ---
try:
    from astropy.io import fits  # type: ignore
    _HAS_ASTROPY = True
except Exception:
    _HAS_ASTROPY = False

# --- Optional: SEP (Source Extractor for Python) ---
try:
    import sep  # type: ignore
    _HAS_SEP = True
except Exception:
    _HAS_SEP = False

# --- Внутренние модули ---
from astroclassify.core.device import pick_device
from astroclassify.api.photometry import (
    has_real_photometry,
    measure_brightness,
    simple_brightness,
)

# -----------------------------------------------------------------------------
# App / logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("astroclassify")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
app = FastAPI(title="AstroClassify API", version="1.2")

# -----------------------------------------------------------------------------
# Limits / security
# -----------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 64 * 1024 * 1024))  # 64 MB
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 80_000_000))       # 80 MP
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FITS_EXTS = {".fits", ".fit", ".fts"}

# -----------------------------------------------------------------------------
# Photometry defaults from ENV
# -----------------------------------------------------------------------------
DEFAULT_R     = float(os.environ.get("ASTRO_PHOT_R",   "3.0"))
DEFAULT_R_IN  = float(os.environ.get("ASTRO_PHOT_RIN", "5.0"))
DEFAULT_R_OUT = float(os.environ.get("ASTRO_PHOT_ROUT","8.0"))

# -----------------------------------------------------------------------------
# Prometheus metrics
# -----------------------------------------------------------------------------
def _build_registry() -> CollectorRegistry:
    registry = CollectorRegistry()
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        multiprocess.MultiProcessCollector(registry)
    return registry

REGISTRY = _build_registry()

REQ_COUNTER = Counter(
    "astro_requests_total", "Total requests", ["endpoint", "status"], registry=REGISTRY
)
REQ_LATENCY = Histogram(
    "astro_request_seconds", "Request latency (s)", ["endpoint"], registry=REGISTRY
)
BYTES_IN = Histogram(
    "astro_upload_size_bytes", "Upload size", ["endpoint"], registry=REGISTRY
)
CLASSIFY_TOPK = Gauge(
    "astro_classify_topk", "Top-K requested in classification", registry=REGISTRY
)
GPU_OOM_COUNTER = Counter(
    "astro_gpu_oom_total", "GPU OOM occurrences", registry=REGISTRY
)

# -----------------------------------------------------------------------------
# Concurrency limiter for inference
# -----------------------------------------------------------------------------
_MAX_INF = int(os.environ.get("ASTRO_MAX_CONCURRENT_INFERENCES", "2"))
_infer_sema = threading.BoundedSemaphore(max(1, _MAX_INF))

@contextmanager
def _infer_slot():
    _infer_sema.acquire()
    try:
        yield
    finally:
        _infer_sema.release()

# -----------------------------------------------------------------------------
# Torch model (ImageNet) with lazy init + locking
# -----------------------------------------------------------------------------
_model_lock = threading.Lock()
_imagenet_model: Optional[nn.Module] = None
_imagenet_classes: Optional[List[str]] = None
_device_str = None


def _ensure_model(device_str: Optional[str] = None) -> Tuple[nn.Module, List[str], str]:
    """Ленивая инициализация torchvision-модели и классов ImageNet."""
    global _imagenet_model, _imagenet_classes, _device_str
    with _model_lock:
        if _imagenet_model is not None and _imagenet_classes is not None:
            return _imagenet_model, _imagenet_classes, _device_str or "cpu"

        picked = pick_device(prefer_cuda=True)
        _device_str = device_str or picked
        logger.info(f"Device picked: {_device_str}")

        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            classes = list(weights.meta["categories"])
            model.eval()
        except Exception:
            logger.exception("Failed to load torchvision model")
            raise HTTPException(status_code=500, detail="Model load error")

        try:
            if _device_str.startswith("cuda") and torch.cuda.is_available():
                model.to(torch.device(_device_str))
            else:
                _device_str = "cpu"
                model.to(torch.device("cpu"))
        except Exception as e:
            logger.warning(f"CUDA move failed ({e}); falling back to CPU.")
            _device_str = "cpu"
            model.to(torch.device("cpu"))

        _imagenet_model = model
        _imagenet_classes = classes
        return model, classes, _device_str


def _handle_cuda_oom(e: RuntimeError) -> None:
    msg = str(e).lower()
    if "out of memory" in msg or "no kernel image is available" in msg:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        GPU_OOM_COUNTER.inc()
        raise HTTPException(status_code=503, detail="GPU is out of memory for this request")
    raise

# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def _safe_basename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w.\-+]+", "_", name)
    return name or "upload.bin"


def _content_disposition_inline(filename: str) -> Dict[str, str]:
    quoted = urllib.parse.quote(filename)
    return {"Content-Disposition": f"inline; filename*=UTF-8''{quoted}"}


def _open_pil_image(data: bytes) -> Image.Image:
    """Безопасное открытие PIL (учёт DecompressionBomb и т.п.)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            img = Image.open(io.BytesIO(data))
            img.load()  # принудительно загрузить
            return img
    except ImageDecompressionBombError as e:
        raise HTTPException(status_code=413, detail=f"Image too large: {e}")
    except Warning as w:  # DecompressionBombWarning → как ошибка
        raise HTTPException(status_code=413, detail=f"Image warning: {w}")
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Unsupported image: {e}")


def _pil_to_array_gray(img: Image.Image) -> np.ndarray:
    """PIL -> grayscale float32 array, contiguous."""
    if img.mode not in ("L", "I;16", "I", "F", "RGB", "RGBA"):
        img = img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGB", "RGBA"):
        img = img.convert("L")
    else:
        img = img.convert("L")
    arr = np.array(img, dtype=np.float32, copy=False)
    return np.ascontiguousarray(arr)


def _to_display_image(arr_or_img: Any) -> Image.Image:
    """Унифицированный превью-рендер: принимается PIL.Image или H×W(×C) массив, возвращается RGB."""
    if isinstance(arr_or_img, Image.Image):
        img = arr_or_img
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    arr = np.asarray(arr_or_img)
    if arr.ndim == 2:
        a_min = float(np.nanmin(arr))
        a_max = float(np.nanmax(arr))
        denom = (a_max - a_min) if a_max > a_min else 1.0
        vis = ((arr - a_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(vis, mode="L").convert("RGB")
    elif arr.ndim == 3:
        # H×W×C → среднее по каналу, ось 2 (исправление прежней регрессии axis=3)
        if arr.shape[2] > 1:
            arr = arr.mean(axis=2)
        a_min = float(np.nanmin(arr))
        a_max = float(np.nanmax(arr))
        denom = (a_max - a_min) if a_max > a_min else 1.0
        vis = ((arr - a_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(vis, mode="L").convert("RGB")
    else:
        raise HTTPException(status_code=500, detail="Unsupported array shape for preview")


def _read_upload_enforced(file: UploadFile, endpoint_name: str) -> Tuple[np.ndarray, str]:
    """
    Универсальный ридер:
    - FITS → H×W float32 (если astropy есть)
    - JPG/PNG/BMP/WEBP/TIFF → L float32
    Ограничивает размер и проверяет пиксели.
    Логирует BYTES_IN с меткой эндпоинта.
    """
    filename = _safe_basename(file.filename or "upload.bin")
    ext = os.path.splitext(filename)[1].lower()

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    BYTES_IN.labels(endpoint=endpoint_name).observe(len(raw))
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    if ext in FITS_EXTS:
        if not _HAS_ASTROPY:
            raise HTTPException(status_code=415, detail="FITS requires astropy")
        try:
            with fits.open(io.BytesIO(raw), memmap=False) as hdul:
                hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                if hdu is None:
                    raise HTTPException(status_code=415, detail="No image data in FITS")
                data = np.array(hdu.data, dtype=np.float32, copy=False)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                if data.ndim == 3:
                    if data.shape[0] <= 4 and data.shape[0] < min(data.shape[1:]):
                        data = data.mean(axis=0)
                    elif data.shape[-1] <= 4:
                        data = data.mean(axis=2)
                    else:
                        data = data[data.shape[0] // 2, :, :]
                if data.ndim != 2:
                    raise HTTPException(status_code=415, detail="FITS must be 2D image")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=415, detail=f"FITS read error: {e}")

    elif ext in IMG_EXTS:
        try:
            img = _open_pil_image(raw)
            data = _pil_to_array_gray(img)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=415, detail=f"Image read error: {e}")
    else:
        # Попытка как изображение по содержимому
        try:
            img = _open_pil_image(raw)
            data = _pil_to_array_gray(img)
        except Exception:
            if _HAS_ASTROPY:
                try:
                    with fits.open(io.BytesIO(raw), memmap=False) as hdul:
                        hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                        if hdu is None:
                            raise HTTPException(status_code=415, detail="No image data")
                        data = np.array(hdu.data, dtype=np.float32, copy=False)
                        if data.ndim == 3:
                            if data.shape[0] <= 4 and data.shape[0] < min(data.shape[1:]):
                                data = data.mean(axis=0)
                            elif data.shape[-1] <= 4:
                                data = data.mean(axis=2)
                            else:
                                data = data[data.shape[0] // 2, :, :]
                        if data.ndim != 2:
                            raise HTTPException(status_code=415, detail="FITS must be 2D")
                except Exception:
                    raise HTTPException(status_code=415, detail="Unsupported file type")
            else:
                raise HTTPException(status_code=415, detail="Unsupported file type")

    h, w = int(data.shape[0]), int(data.shape[1])
    if (h * w) > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image has too many pixels")

    return np.ascontiguousarray(data.astype(np.float32, copy=False)), filename


# -----------------------------------------------------------------------------
# Health / metrics
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> Response:
    return PlainTextResponse("ok")


@app.get("/metrics")
def metrics() -> Response:
    try:
        content = generate_latest(REGISTRY)
        return Response(content, media_type=CONTENT_TYPE_LATEST)
    except Exception:
        logger.exception("Prometheus metrics error")
        raise HTTPException(status_code=500, detail="metrics error")


# -----------------------------------------------------------------------------
# Classification endpoints (ImageNet demo)
# -----------------------------------------------------------------------------
_IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

def _pil_from_array_for_classify(arr: np.ndarray) -> Image.Image:
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    denom = (a_max - a_min) if a_max > a_min else 1.0
    vis = ((arr - a_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(vis, mode="L").convert("RGB")
    return img

def _maybe_imagenet_norm(t: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return t
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std

@app.post("/classify")
def classify_endpoint(
    file: UploadFile = File(...),
    topk: int = Query(5, ge=1, le=1000),
    imagenet_norm: bool = Query(True),
):
    tic = time.time()
    endpoint = "/classify"
    tensor = logits = None
    try:
        data, filename = _read_upload_enforced(file, endpoint_name=endpoint)
        img = _pil_from_array_for_classify(data)

        model, classes, device_str = _ensure_model()
        CLASSIFY_TOPK.set(topk)

        tensor = _IMAGENET_TRANSFORM(img)
        tensor = _maybe_imagenet_norm(tensor, imagenet_norm)
        tensor = tensor.unsqueeze(0)

        with _infer_slot():
            dev = torch.device(device_str if device_str != "cpu" else "cpu")
            tensor = tensor.to(dev)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_idx = probs.argsort()[::-1][:topk]
        results = [
            {"rank": int(i + 1), "class": classes[idx], "prob": float(probs[idx])}
            for i, idx in enumerate(top_idx)
        ]
        payload = {
            "filename": filename,
            "device": device_str,
            "topk": topk,
            "results": results,
        }
        REQ_COUNTER.labels(endpoint=endpoint, status="200").inc()
        return JSONResponse(payload)

    except RuntimeError as e:
        _handle_cuda_oom(e)
    except HTTPException:
        REQ_COUNTER.labels(endpoint=endpoint, status="http_error").inc()
        raise
    except Exception:
        REQ_COUNTER.labels(endpoint=endpoint, status="500").inc()
        logger.exception("classify failed")
        raise HTTPException(status_code=500, detail="classification failed")
    finally:
        if isinstance(tensor, torch.Tensor):
            del tensor
        if logits is not None:
            del logits
        REQ_LATENCY.labels(endpoint=endpoint).observe(time.time() - tic)

@app.post("/classify_batch")
def classify_batch_endpoint(
    files: List[UploadFile] = File(...),
    topk: int = Query(5, ge=1, le=1000),
    imagenet_norm: bool = Query(True),
):
    tic = time.time()
    endpoint = "/classify_batch"
    results: List[Dict[str, Any]] = []
    try:
        model, classes, device_str = _ensure_model()
        CLASSIFY_TOPK.set(topk)
        dev = torch.device(device_str if device_str != "cpu" else "cpu")

        for f in files:
            tensor = logits = None
            try:
                data, filename = _read_upload_enforced(f, endpoint_name=endpoint)
                img = _pil_from_array_for_classify(data)
                tensor = _IMAGENET_TRANSFORM(img)
                tensor = _maybe_imagenet_norm(tensor, imagenet_norm)
                tensor = tensor.unsqueeze(0)

                with _infer_slot():
                    tensor = tensor.to(dev)
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                top_idx = probs.argsort()[::-1][:topk]
                one = {
                    "filename": filename,
                    "device": device_str,
                    "topk": topk,
                    "results": [
                        {"rank": int(i + 1), "class": classes[idx], "prob": float(probs[idx])}
                        for i, idx in enumerate(top_idx)
                    ],
                }
                results.append(one)
            except RuntimeError as e:
                try:
                    _handle_cuda_oom(e)
                except HTTPException as he:
                    results.append({"filename": f.filename, "error": he.detail})
            except Exception:
                results.append({"filename": f.filename, "error": "classification failed"})
            finally:
                if isinstance(tensor, torch.Tensor):
                    del tensor
                if logits is not None:
                    del logits

        payload = {"count": len(results), "results": results}
        REQ_COUNTER.labels(endpoint=endpoint, status="200").inc()
        return JSONResponse(payload)

    except HTTPException:
        REQ_COUNTER.labels(endpoint=endpoint, status="http_error").inc()
        raise
    except Exception:
        REQ_COUNTER.labels(endpoint=endpoint, status="500").inc()
        logger.exception("classify_batch failed")
        raise HTTPException(status_code=500, detail="classification failed")
    finally:
        REQ_LATENCY.labels(endpoint=endpoint).observe(time.time() - tic)

# -----------------------------------------------------------------------------
# Helper: normalize photometry output to List[float]
# -----------------------------------------------------------------------------
def _extract_flux_list(fluxes) -> List[float]:
    """
    Допускает разные форматы из measure_brightness:
    - [float, ...]
    - [{"flux": x} ...] или [{"aperture_sum": x} ...]
    - и пр. варианты — берём наиболее вероятные ключи
    """
    out: List[float] = []
    if fluxes is None:
        return out
    for f in fluxes:
        if isinstance(f, (int, float)):
            out.append(float(f))
        elif isinstance(f, dict):
            for key in ("flux", "aperture_sum", "sum", "flux_total", "value"):
                val = f.get(key, None)
                if isinstance(val, (int, float)):
                    out.append(float(val))
                    break
            else:
                # fallback
                try:
                    out.append(float(next(iter(f.values()))))
                except Exception:
                    out.append(0.0)
        else:
            try:
                out.append(float(f))
            except Exception:
                out.append(0.0)
    return out

# -----------------------------------------------------------------------------
# SEP detection endpoint (improved) + proper photometry call
# -----------------------------------------------------------------------------
@app.post("/detect_auto")
def detect_auto_endpoint(
    file: UploadFile = File(...),
    detector: str = Query("sep"),
    threshold_sigma: float = Query(1.5, ge=0.1, le=50.0),
    topk: int = Query(10, ge=1, le=10_000),
    background_subtract: bool = Query(True),
    preview: bool = Query(False),
    # Новые параметры для апертурной фотометрии, из ENV по умолчанию:
    r: float = Query(DEFAULT_R, ge=0.5, le=1000.0),
    r_in: float = Query(DEFAULT_R_IN, ge=0.5, le=2000.0),
    r_out: float = Query(DEFAULT_R_OUT, ge=0.5, le=4000.0),
):
    """
    Автоматическое выделение источников SEP + опциональная апертурная фотометрия.
    - preview=true → вернёт JPEG с оверлеем
    """
    if detector.lower() != "sep":
        raise HTTPException(status_code=400, detail="Only detector=sep is supported")
    if not _HAS_SEP:
        raise HTTPException(status_code=501, detail="sep is not installed")

    tic = time.time()
    endpoint = "/detect_auto"
    try:
        data, filename = _read_upload_enforced(file, endpoint_name=endpoint)
        data = np.ascontiguousarray(data.astype(np.float32, copy=False))

        # === Background estimation ===
        if background_subtract:
            bkg = sep.Background(data)
            data_sub = data - bkg.back()
        else:
            data_sub = data

        # === Detection ===
        std = float(np.std(data_sub))
        if not math.isfinite(std) or std <= 0:
            std = 1.0
        thresh = threshold_sigma * std

        objects = sep.extract(
            data_sub,
            thresh=thresh,
            err=None,
            mask=None,
            minarea=8,
            deblend_nthresh=32,
            deblend_cont=0.005,
            filter_kernel=np.ones((3, 3), dtype=np.uint8),
            clean=True,
        )
        if objects is None or len(objects) == 0:
            raise HTTPException(status_code=404, detail="No sources detected")

        # === Filtering ===
        objects = objects[objects["flag"] == 0]
        if len(objects) == 0:
            raise HTTPException(status_code=404, detail="No clean sources detected")

        if "flux" in objects.dtype.names and len(objects) > 20:
            flux_cut = np.percentile(objects["flux"], 5.0)
            objects = objects[objects["flux"] > flux_cut]
            if len(objects) == 0:
                raise HTTPException(status_code=404, detail="No sources after flux cut")

        # === Sort and topk ===
        if "flux" in objects.dtype.names:
            objects = np.sort(objects, order="flux")[::-1]
        if topk and len(objects) > topk:
            objects = objects[:topk]

        positions = np.column_stack((objects["x"], objects["y"]))
        result = {
            "filename": filename,
            "mode": "auto-aperture",
            "detector": "sep",
            "real_photometry": has_real_photometry(),
            "count": int(len(objects)),
            "positions": [{"x": float(x), "y": float(y)} for x, y in positions],
            "threshold_sigma": float(threshold_sigma),
            "background_subtract": bool(background_subtract),
            "r": float(r),
            "r_in": float(r_in),
            "r_out": float(r_out),
        }

        # === Photometry integration (корректный вызов с радиусами) ===
        try:
            phot = measure_brightness(
                data,
                positions=positions,
                r=float(r),
                r_in=float(r_in),
                r_out=float(r_out),
            )

            # Normalise/normalize the photometry output into a list of numeric fluxes.
            extracted_fluxes: List[float] = []

            # Case: single numeric (fallback simple_brightness)
            if isinstance(phot, (int, float, np.integer, np.floating)):
                # replicate the scalar for all positions (best-effort)
                extracted_fluxes = [float(phot)] * len(positions)

            # Case: list-like
            elif isinstance(phot, list):
                if len(phot) == 0:
                    extracted_fluxes = []
                else:
                    # list of dicts (preferred for real photometry)
                    if isinstance(phot[0], dict):
                        for item in phot:
                            if not isinstance(item, dict):
                                # try to coerce
                                try:
                                    extracted_fluxes.append(float(item))
                                except Exception:
                                    extracted_fluxes.append(0.0)
                                continue

                            if "flux_sub" in item:
                                extracted_fluxes.append(float(item["flux_sub"]))
                            elif "aperture_sum" in item:
                                extracted_fluxes.append(float(item["aperture_sum"]))
                            else:
                                # last resort: try to find any numeric value
                                found = False
                                for k in ("flux", "flux_sub", "aperture_sum"):
                                    if k in item:
                                        extracted_fluxes.append(float(item[k]))
                                        found = True
                                        break
                                if not found:
                                    # fall back to 0.0
                                    extracted_fluxes.append(0.0)

                    else:
                        # list of numbers
                        for v in phot:
                            try:
                                extracted_fluxes.append(float(v))
                            except Exception:
                                extracted_fluxes.append(0.0)

            else:
                # unknown type — leave as empty
                extracted_fluxes = []

            # Attach fluxes to positions conservatively (check lengths)
            for i, f in enumerate(extracted_fluxes):
                if i < len(result["positions"]):
                    result["positions"][i]["flux"] = float(f)

        except Exception as e:
            result["warning"] = f"Photometry failed: {e}"

        # === Preview overlay ===
        if preview:
            img = _to_display_image(data)
            draw = ImageDraw.Draw(img)
            # objects is a numpy recarray — use dtype names to access optional fields
            has_a = "a" in objects.dtype.names
            for obj in objects:
                rr = max(2.0, float(obj["a"]) if has_a else 3.0)
                x, y = float(obj["x"]), float(obj["y"])
                draw.ellipse((x - rr, y - rr, x + rr, y + rr), outline="red", width=1)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            headers = _content_disposition_inline(f"{os.path.splitext(filename)[0]}_sep.jpg")
            REQ_COUNTER.labels(endpoint=endpoint, status="200").inc()
            REQ_LATENCY.labels(endpoint=endpoint).observe(time.time() - tic)
            return StreamingResponse(buf, media_type="image/jpeg", headers=headers)

        REQ_COUNTER.labels(endpoint=endpoint, status="200").inc()
        return JSONResponse(result)

    except HTTPException:
        REQ_COUNTER.labels(endpoint=endpoint, status="http_error").inc()
        raise
    except Exception:
        REQ_COUNTER.labels(endpoint=endpoint, status="500").inc()
        logger.exception("detect_auto failed")
        raise HTTPException(status_code=500, detail="detection failed")
    finally:
        REQ_LATENCY.labels(endpoint=endpoint).observe(time.time() - tic)

# -----------------------------------------------------------------------------
# Simple brightness endpoint (no heavy deps)
# -----------------------------------------------------------------------------
@app.post("/brightness_simple")
def brightness_simple(
    file: UploadFile = File(...),
):
    """Лёгкая оценка яркости (без photutils), на базе simple_brightness()."""
    endpoint = "/brightness_simple"
    try:
        data, filename = _read_upload_enforced(file, endpoint_name=endpoint)
        val = float(simple_brightness(data))
        return JSONResponse({"filename": filename, "brightness": val})
    except Exception:
        logger.exception("brightness_simple failed")
        raise HTTPException(status_code=500, detail="brightness failed")

# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
def root() -> Response:
    return PlainTextResponse(
        "AstroClassify API is up. "
        "/health /metrics /classify /classify_batch /detect_auto /brightness_simple"
    )

# -----------------------------------------------------------------------------
# Optional: preload model at startup for production
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _maybe_preload():
    if os.environ.get("ASTRO_PRELOAD_MODEL", "0") in ("1", "true", "yes"):
        try:
            _ensure_model()
            logger.info("Model preloaded on startup")
        except Exception:
            logger.exception("Failed to preload model on startup")
