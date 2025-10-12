# api/main.py
from __future__ import annotations

import io, os, time
from typing import List, Dict, Any

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# ── Конфиг ─────────────────────────────────────────────────────────────────────
CLASSES = ["star", "galaxy", "artifact", "transient"]
IMG_SIZE = 224
TOPK = 3

# Ограничения ввода
MAX_UPLOAD_SIZE = int(os.getenv("ASTRO_MAX_UPLOAD", 10 * 1024 * 1024))  # 10 MiB по умолчанию
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/jpg"}

# CPU-режим (до появления официальной поддержки SM 12.0 в torch)
USE_AMP = False
device = "cpu"

# Тюнинг тредов под EPYC (перекрывается переменными окружения)
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "8"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("MKL_NUM_THREADS", "8"))
torch.set_num_threads(int(os.getenv("ASTRO_TORCH_THREADS", "8")))
torch.set_num_interop_threads(int(os.getenv("ASTRO_TORCH_INTEROP", "2")))
torch.backends.mkldnn.enabled = True

# Нормализация
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="AstroClassify API", version="0.3.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Модель ─────────────────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.head  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # 112x112
        x = self.pool(F.relu(self.conv2(x)))  # 56x56
        x = F.relu(self.conv3(x))             # 56x56
        return self.head(x)

_base = TinyCNN(num_classes=len(CLASSES)).eval()
try:
    model: nn.Module = torch.quantization.quantize_dynamic(_base, {nn.Linear}, dtype=torch.qint8)
except Exception:
    model = _base
model = model.to(device).eval()
try:
    model = model.to(memory_format=torch.channels_last)
except Exception:
    pass

# ── Вспомогательные функции ────────────────────────────────────────────────────
def _choose_resample(orig_w: int, orig_h: int, target: int) -> int:
    # LANCZOS лучше для даунскейла; BICUBIC — универсальный.
    try:
        from PIL import Image as _Image
        if max(orig_w, orig_h) > target:
            return _Image.Resampling.LANCZOS
        return _Image.Resampling.BICUBIC
    except Exception:
        return Image.BICUBIC

def pil_to_tensor(img: Image.Image, size: int = IMG_SIZE, imagenet_norm: bool = True) -> torch.Tensor:
    """PIL -> torch (NCHW), корректно применяем channels_last ТОЛЬКО для 4D; адаптивная интерполяция."""
    w, h = img.size
    resample = _choose_resample(w, h, size)
    img = img.convert("RGB").resize((size, size), resample=resample)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if imagenet_norm:
        arr = (arr - np.array(NORM_MEAN, dtype=np.float32)) / np.array(NORM_STD, dtype=np.float32)

    # CHW -> NCHW
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)

    # channels_last допустим только для 4D (NCHW)
    try:
        if x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)
    except Exception:
        pass

    return x.to(device)

def softmax_topk(logits: torch.Tensor, k: int = TOPK) -> List[Dict[str, float]]:
    probs = torch.softmax(logits, dim=1)
    k = min(k, probs.shape[1])
    v, i = torch.topk(probs, k=k, dim=1)
    return [{"label": CLASSES[int(ii)], "p": float(vv)} for vv, ii in zip(v[0].tolist(), i[0].tolist())]

def image_long_side_px(img: Image.Image) -> int:
    w, h = img.size
    return max(w, h)

def _looks_like_image(buf: bytes) -> bool:
    # Грубая проверка заголовков — jpeg/png/webp/tiff
    sig = buf[:16]
    return (
        sig.startswith(b"\xff\xd8") or
        sig.startswith(b"\x89PNG\r\n\x1a\n") or
        (sig.startswith(b"RIFF") and b"WEBP" in buf[:32]) or
        sig[:4] in (b"II*\x00", b"MM\x00*")
    )

# ── Prometheus ──────────────────────────────────────────────────────────────────
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest

REG = CollectorRegistry(auto_describe=True)

REQ_COUNTER = Counter(
    "astro_requests_total", "Total HTTP requests",
    labelnames=("endpoint", "status", "device"), registry=REG,
)
LAT_HIST = Histogram(
    "astro_request_latency_ms", "Request latency (ms)",
    labelnames=("endpoint", "device"),
    buckets=(50, 100, 200, 400, 800, 1600, 3200, 6400, 10000),
    registry=REG,
)
IMG_SIZE_HIST = Histogram(
    "astro_image_size_px", "Input image long side (px)",
    labelnames=("endpoint",), buckets=(224, 512, 1024, 2048, 4096, 8192, 16384),
    registry=REG,
)
BATCH_HIST = Histogram(
    "astro_batch_images", "Batch size for /classify_batch",
    buckets=(1, 2, 4, 8, 16, 32, 64), registry=REG,
)
FALLBACKS = Counter(
    "astro_infer_fallback_total", "CUDA→CPU fallbacks by reason",
    labelnames=("reason",), registry=REG,
)
CPU_THREADS = Gauge("astro_cpu_threads", "Torch/BLAS threads", labelnames=("type",), registry=REG)
CPU_THREADS.labels(type="torch_num_threads").set(torch.get_num_threads())
CPU_THREADS.labels(type="torch_interop_threads").set(torch.get_num_interop_threads())
CPU_THREADS.labels(type="omp_num_threads").set(int(os.environ.get("OMP_NUM_THREADS", "0")))
CPU_THREADS.labels(type="mkl_num_threads").set(int(os.environ.get("MKL_NUM_THREADS", "0")))

# Метрики по загрузкам
UPLOAD_BYTES = Histogram(
    "astro_upload_bytes",
    "Uploaded file size in bytes",
    labelnames=("endpoint",),
    buckets=(128*1024, 512*1024, 1*1024*1024, 2*1024*1024, 5*1024*1024, 10*1024*1024, 20*1024*1024),
    registry=REG,
)
UPLOAD_OVERSIZE = Counter(
    "astro_upload_oversize_total",
    "Uploads rejected due to size limit",
    labelnames=("endpoint",),
    registry=REG,
)
UNSUPPORTED_MIME = Counter(
    "astro_unsupported_mime_total",
    "Uploads rejected due to unsupported content type",
    labelnames=("endpoint", "content_type"),
    registry=REG,
)

# ── Middleware для метрик ──────────────────────────────────────────────────────
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.time()
        endpoint = request.url.path
        status = "500"
        try:
            resp = await call_next(request)
            status = str(getattr(resp, "status_code", 200))
            return resp
        except HTTPException as he:
            status = str(he.status_code)
            raise
        finally:
            if endpoint in ("/classify", "/classify_batch", "/photometry"):
                LAT_HIST.labels(endpoint=endpoint, device=device).observe((time.time()-t0)*1000.0)
                REQ_COUNTER.labels(endpoint=endpoint, status=status, device=device).inc()

app.add_middleware(MetricsMiddleware)

# ── Валидация загрузки ─────────────────────────────────────────────────────────
def _validate_upload(file: UploadFile, content: bytes, endpoint: str) -> bool:
    size = len(content or b"")
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty upload")

    # наблюдаем размер
    try:
        UPLOAD_BYTES.labels(endpoint=endpoint).observe(size)
    except Exception:
        pass

    # лимит
    if size > MAX_UPLOAD_SIZE:
        try:
            UPLOAD_OVERSIZE.labels(endpoint=endpoint).inc()
        except Exception:
            pass
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_SIZE} bytes)")

    # MIME
    ctype = (file.content_type or "").lower()
    if ctype and (ctype not in ALLOWED_IMAGE_TYPES):
        try:
            UNSUPPORTED_MIME.labels(endpoint=endpoint, content_type=ctype).inc()
        except Exception:
            pass
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {ctype}")

    # Если тип пустой — дадим шанс PIL, но слегка проверим сигнатуру
    if not ctype and not _looks_like_image(content):
        raise HTTPException(status_code=415, detail="Unsupported media: unrecognized signature")

    return True

# ── Эндпоинты ───────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health", operation_id="health")
def health() -> Dict[str, Any]:
    torch_cuda = getattr(torch.version, "cuda", "") or ""
    return {
        "status": "ok", "device": device, "classes": CLASSES,
        "amp": False, "model": model.__class__.__name__, "torch_cuda": torch_cuda
    }

@app.get("/metrics", summary="Metrics", operation_id="metrics")
def metrics() -> Response:
    return Response(content=generate_latest(REG), media_type=CONTENT_TYPE_LATEST)

@app.post("/classify", summary="Classify one image", operation_id="classify")
async def classify(
    file: UploadFile,
    topk: int = Query(TOPK, ge=1, le=len(CLASSES)),
    imagenet_norm: bool = Query(True),
):
    try:
        content = await file.read()
        _validate_upload(file, content, "/classify")

        try:
            img = Image.open(io.BytesIO(content))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Unsupported image format")

        IMG_SIZE_HIST.labels(endpoint="/classify").observe(image_long_side_px(img))

        x = pil_to_tensor(img, imagenet_norm=imagenet_norm)
        with torch.no_grad():
            logits = model(x)

        return {"top1": softmax_topk(logits, 1)[0], "topk": softmax_topk(logits, topk)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}")

@app.post("/classify_batch", summary="Classify a batch", operation_id="classify_batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    topk: int = Query(TOPK, ge=1, le=len(CLASSES)),
    imagenet_norm: bool = Query(True),
):
    out: List[Dict[str, Any]] = []
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files")
        BATCH_HIST.observe(len(files))

        for f in files:
            try:
                content = await f.read()
                _validate_upload(f, content, "/classify_batch")

                img = Image.open(io.BytesIO(content))
                IMG_SIZE_HIST.labels(endpoint="/classify_batch").observe(image_long_side_px(img))

                x = pil_to_tensor(img, imagenet_norm=imagenet_norm)
                with torch.no_grad():
                    logits = model(x)

                out.append({
                    "filename": f.filename,
                    "top1": softmax_topk(logits, 1)[0],
                    "topk": softmax_topk(logits, topk),
                })
            except HTTPException as he:
                out.append({"filename": f.filename, "error": he.detail})
            except UnidentifiedImageError:
                out.append({"filename": f.filename, "error": "Unsupported image format"})
            except Exception as e:
                out.append({"filename": f.filename, "error": str(e)})

        return {"count": len(out), "results": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing error: {e}")

# ── Фотометрия ─────────────────────────────────────────────────────────────────
def simple_photometry_gray(arr: np.ndarray, x: int, y: int, r: float = 6.0) -> Dict[str, float]:
    # Ленивый импорт photutils
    try:
        from photutils.aperture import CircularAperture, aperture_photometry
    except Exception:
        # Понятная ошибка для пользователя/оператора
        raise HTTPException(
            status_code=500,
            detail="Photometry unavailable: install 'photutils' (e.g. pip install photutils)"
        )

    ap = CircularAperture([(x, y)], r=r)
    tbl = aperture_photometry(arr, ap)
    flux = float(tbl["aperture_sum"][0])

    # Примитивная оценка фона/шума
    bkg = float(np.median(arr))
    area = float(np.pi * r * r)
    net_flux = float(max(flux - bkg * area, 0.0))

    # Устойчивый SNR: snr ≈ net_flux / sqrt(flux + eps)
    eps = 1e-6
    denom = float(np.sqrt(max(flux, eps)))
    snr = net_flux / (denom + eps)

    return {"flux": flux, "background": bkg, "snr": snr}

@app.post("/photometry", summary="Aperture photometry (mono)", operation_id="photometry")
async def photometry(
    file: UploadFile,
    x: int = Query(..., ge=0),
    y: int = Query(..., ge=0),
    r: float = Query(6.0, gt=0),
):
    try:
        content = await file.read()
        _validate_upload(file, content, "/photometry")

        img = Image.open(io.BytesIO(content)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w = arr.shape
        if not (0 <= x < w and 0 <= y < h):
            raise HTTPException(status_code=400, detail=f"Point ({x},{y}) out of bounds ({w}x{h})")
        return simple_photometry_gray(arr, x, y, r)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Photometry error: {e}")

# ── Подсказки запуска ──────────────────────────────────────────────────────────
# CPU (ручной):
#   CUDA_VISIBLE_DEVICES="" python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# Gunicorn:
#   ASTRO_TORCH_THREADS=8 ASTRO_TORCH_INTEROP=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
#   gunicorn api.main:app -k uvicorn.workers.UvicornWorker --workers 12 --bind 0.0.0.0:8000 --timeout 60
# Prometheus:
#   /metrics (job: astroclassify)
