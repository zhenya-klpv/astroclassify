# api/main.py
from __future__ import annotations

import io
import os
import time
import json
import csv
import logging
from typing import List, Dict, Any, Iterable

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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

# Мягкая зависимость для автодетекта источников и FITS-экспорта
try:
    from photutils.detection import DAOStarFinder  # type: ignore
except Exception:
    DAOStarFinder = None  # если нет photutils.detection — эндпоинт вернёт 501

try:
    from astropy.table import Table  # type: ignore
except Exception:
    Table = None  # FITS экспорт будет недоступен

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
# FastAPI app — создаём сразу!
# -----------------------------------------------------------------------------
app = FastAPI(title="AstroClassify API", version="0.5.0")

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
    "Total requests",
    ["endpoint", "method", "status"],
    registry=PROM_REGISTRY,
)

LATENCY_HIST = Histogram(
    "astro_request_latency_seconds",
    "Request latency",
    ["endpoint", "method"],
    registry=PROM_REGISTRY,
)

# Новые метрики под фотометрию и детекции
PHOT_COUNTER = Counter(
    "astro_photometry_requests_total",
    "Photometry/detection requests",
    ["endpoint"],
    registry=PROM_REGISTRY,
)
SOURCES_COUNTER = Counter(
    "astro_sources_detected_total",
    "Total sources detected",
    registry=PROM_REGISTRY,
)

def _track(endpoint: str, method: str):
    start = time.perf_counter()

    class _Tracker:
        def ok(self, status: int = 200):
            LATENCY_HIST.labels(endpoint, method).observe(time.perf_counter() - start)
            REQ_COUNTER.labels(endpoint, method, str(status)).inc()

        def fail(self, status: int):
            LATENCY_HIST.labels(endpoint, method).observe(time.perf_counter() - start)
            REQ_COUNTER.labels(endpoint, method, str(status)).inc()

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
_PHOT_FIELDS = ["x", "y", "r", "aperture_sum", "bkg_mean", "bkg_area", "flux_sub"]

def _rows_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        rows.append({k: r.get(k, None) for k in _PHOT_FIELDS})
    return rows

def _serialize_rows(rows: List[Dict[str, Any]], out_format: str) -> tuple[bytes, str, str]:
    """
    Возвращает (data_bytes, media_type, extension)
    out_format: json|csv|fits
    """
    out_format = out_format.lower()
    if out_format == "json":
        data = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
        return data, "application/json", "json"

    if out_format == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_PHOT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue().encode("utf-8"), "text/csv", "csv"

    if out_format == "fits":
        if Table is None:
            raise HTTPException(status_code=501, detail="FITS export requires astropy.table")
        table = Table(rows=rows, names=_PHOT_FIELDS)
        buf = io.BytesIO()
        table.write(buf, format="fits", overwrite=True)
        return buf.getvalue(), "application/fits", "fits"

    raise HTTPException(status_code=400, detail="Unsupported out_format")

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
# Эндпоинты
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

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
        results = _classify_bytes(data, imagenet_norm=imagenet_norm, topk=topk)
        t.ok()
        return {"filename": file.filename, "results": results}
    except Exception as e:
        logger.exception("classify failed")
        t.fail(500)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    topk: int = Query(5, ge=1, le=1000),
    imagenet_norm: bool = Query(True),
):
    t = _track("/classify_batch", "POST")
    out = []
    status = 200
    for f in files:
        try:
            data = await f.read()
            out.append(
                {"filename": f.filename, "results": _classify_bytes(data, imagenet_norm, topk)}
            )
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
    out_format: str = Query("json", pattern="^(json|csv|fits)$"),
    download: bool = Query(False, description="Return as attachment when true"),
):
    """
    Измерение яркости источников.
    - Если заданы `xy` и `r` и доступны astropy+photutils → апертурная фотометрия.
    - Иначе → быстрая оценка яркости (simple_brightness ~ [0..1]).
    Поддерживает экспорт: json|csv|fits.
    """
    t = _track("/detect_sources", "POST")
    try:
        data = await file.read()

        # Парсим координаты "x,y"
        positions: List[tuple[float, float]] = []
        for item in xy:
            try:
                sx, sy = item.split(",", 1)
                positions.append((float(sx), float(sy)))
            except Exception:
                t.fail(400)
                raise HTTPException(status_code=400, detail=f"Bad xy value: {item!r}")

        do_aperture = bool(positions and r is not None and has_real_photometry())

        if do_aperture:
            results = measure_brightness(
                data, positions=positions, r=float(r), r_in=r_in, r_out=r_out
            )

            # метрики
            PHOT_COUNTER.labels("/detect_sources").inc()
            SOURCES_COUNTER.inc(len(positions))

            # экспорт
            if out_format != "json":
                rows = _rows_from_results(results)  # type: ignore[arg-type]
                buf, media, ext = _serialize_rows(rows, out_format)
                filename = (file.filename or "image") + f".photometry.{ext}"
                headers = {"Content-Disposition": f'{"attachment" if download else "inline"}; filename="{filename}"'}
                t.ok()
                return Response(content=buf, media_type=media, headers=headers)

            payload = {
                "filename": file.filename,
                "mode": "aperture",
                "real_photometry": True,
                "count": len(positions),
                "results": results,
            }
            t.ok()
            return payload

        # Фоллбек — для simple режима экспорт таблиц не применим
        if out_format != "json":
            t.fail(400)
            raise HTTPException(status_code=400, detail="Export available only for aperture mode")

        val = simple_brightness(data)
        payload = {
            "filename": file.filename,
            "mode": "simple",
            "real_photometry": has_real_photometry(),
            "simple_brightness": val,
        }
        # метрики
        PHOT_COUNTER.labels("/detect_sources").inc()
        SOURCES_COUNTER.inc(0)

        t.ok()
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("detect_sources failed")
        t.fail(500)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_auto")
async def detect_auto(
    file: UploadFile = File(..., description="Image file: JPG/PNG/TIFF/FITS*"),
    detector: str = Query("dao", pattern="^(dao|sep)$", description="Detector backend"),
    fwhm: float = Query(3.0, ge=1.0, description="Approx. stellar FWHM in pixels"),
    threshold_sigma: float = Query(5.0, ge=0.1, description="Detection threshold in σ"),
    max_sources: int = Query(50, ge=1, le=5000, description="Limit number of returned sources"),
    r: float = Query(5.0, ge=1.0, description="Aperture radius (px)"),
    r_in: float | None = Query(8.0, ge=0.0, description="Annulus inner radius (px)"),
    r_out: float | None = Query(12.0, ge=0.0, description="Annulus outer radius (px)"),
    out_format: str = Query("json", pattern="^(json|csv|fits)$"),
    download: bool = Query(False, description="Return as attachment when true"),
):
    """
    Автопоиск источников (DAOStarFinder или SEP) + апертурная фотометрия по найденным центрам.
    Поддерживает экспорт результатов фотометрии: json|csv|fits.
    """
    t = _track("/detect_auto", "POST")

    if not has_real_photometry():
        t.fail(501)
        raise HTTPException(status_code=501, detail="Real photometry (photutils/astropy) is required.")

    if detector == "dao" and DAOStarFinder is None:
        t.fail(501)
        raise HTTPException(status_code=501, detail="DAOStarFinder requires photutils.detection")

    try:
        data = await file.read()

        # Подготавливаем изображение (градации серого + нормализация)
        arr = _to_float_array(data)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        arr = _auto_normalize(arr)

        # Робастная оценка σ фона (MAD → σ)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        robust_sigma = 1.4826 * mad if mad > 0 else float(np.std(arr))
        threshold = threshold_sigma * robust_sigma

        # --- выбор детектора: SEP или DAO ---
        positions: list[tuple[float, float]] = []

        if detector == "sep":
            if sep is None:
                t.fail(501)
                raise HTTPException(status_code=501, detail="SEP is not installed; pip install sep")

            # SEP требует float32 и C-contiguous
            data32 = np.ascontiguousarray(arr.astype(np.float32))
            bkg = sep.Background(data32)
            data_sub = data32 - bkg.back()
            # threshold в абсолютных единицах кадра
            thresh_abs = threshold_sigma * bkg.globalrms

            objects = sep.extract(data_sub, thresh_abs, minarea=5)
            if objects is not None and len(objects) > 0:
                order = np.argsort(objects["flux"])[::-1]
                for idx in order[:max_sources]:
                    positions.append((float(objects["x"][idx]), float(objects["y"][idx])))
        else:
            # DAOStarFinder
            finder = DAOStarFinder(fwhm=fwhm, threshold=threshold)
            tbl = finder(arr - med)  # центрируем на медиане
            if tbl is not None and len(tbl) > 0:
                order = np.argsort(np.array(tbl["flux"]))[::-1]
                for idx in order[:max_sources]:
                    x = float(tbl["xcentroid"][idx])
                    y = float(tbl["ycentroid"][idx])
                    positions.append((x, y))

        # Если ничего не нашли — вернём пустой ответ (json)
        if not positions:
            # метрики
            PHOT_COUNTER.labels("/detect_auto").inc()
            SOURCES_COUNTER.inc(0)

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

        # Фотометрия по найденным центрам
        phot = measure_brightness(
            data, positions=positions, r=float(r), r_in=r_in, r_out=r_out
        )

        # метрики
        PHOT_COUNTER.labels("/detect_auto").inc()
        SOURCES_COUNTER.inc(len(positions))

        # Экспорт таблицы, если требуется
        if out_format != "json":
            rows = _rows_from_results(phot)  # type: ignore[arg-type]
            buf, media, ext = _serialize_rows(rows, out_format)
            filename = (file.filename or "image") + f".auto.photometry.{ext}"
            headers = {"Content-Disposition": f'{"attachment" if download else "inline"}; filename="{filename}"'}
            t.ok()
            return Response(content=buf, media_type=media, headers=headers)

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
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("detect_auto failed")
        t.fail(500)
        raise HTTPException(status_code=500, detail=str(e))

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
):
    """
    Рендерит PNG с контурами апертуры (красный) и аннулуса (зелёный).
    Удобно для визуальной проверки параметров.
    """
    t = _track("/preview_apertures", "POST")
    try:
        data = await file.read()

        # Парсим координаты
        positions: List[tuple[float, float]] = []
        for item in xy:
            try:
                sx, sy = item.split(",", 1)
                positions.append((float(sx), float(sy)))
            except Exception:
                t.fail(400)
                raise HTTPException(status_code=400, detail=f"Bad xy value: {item!r}")

        if not positions:
            t.fail(400)
            raise HTTPException(status_code=400, detail="At least one xy coordinate is required")

        # Декодим исходник
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            overlay = _draw_circles(im, positions, r=r, r_in=r_in, r_out=r_out, line=line)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)

        headers = {"Content-Disposition": f'inline; filename="{(file.filename or "image")}.preview.png"'}
        t.ok()
        return StreamingResponse(buf, media_type="image/png", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("preview_apertures failed")
        t.fail(500)
        raise HTTPException(status_code=500, detail=str(e))
