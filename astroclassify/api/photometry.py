"""Photometry utilities for AstroClassify.

Two modes of operation:

1) **Real aperture photometry** (preferred):
   If `photutils` and `astropy` are available, we do proper circular-aperture
   photometry with optional local background subtraction via a circular annulus.
   ⚠️ Aperture photometry сохраняет исходные счётные единицы пикселей (без нормализации).

2) **Lightweight fallback**:
   If those libs are not installed, we compute a simple brightness estimate
   using a per-pixel mean across channels. This has **no external deps** beyond
   Pillow and NumPy (which the project already uses).

The public API is small and stable:

- `has_real_photometry() -> bool`
- `measure_brightness(image, positions=None, r=None, r_in=None, r_out=None)`
- `simple_brightness(image)`

`image` can be one of:
- A file path (PNG/JPG/TIFF/FITS*)
- `PIL.Image.Image`
- `np.ndarray` with shape (H, W) or (H, W, C)
- Raw bytes (an encoded image); we'll attempt to decode via Pillow

*FITS reading requires astropy; if not present, FITS input will raise.

Example (aperture photometry):

>>> from astroclassify.api.photometry import measure_brightness
>>> result = measure_brightness("example.jpg", positions=[(120.5, 80.2)], r=8,
...                             r_in=10, r_out=15)
>>> print(result[0]["flux_sub"])  # background-subtracted aperture sum

Example (fallback):

>>> from astroclassify.api.photometry import simple_brightness
>>> val = simple_brightness("example.jpg")
>>> print(val)

CLI quick test:

    python -m astroclassify.api.photometry path/to/image.jpg --x 120 --y 80 --r 10
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import io
import math
import os
import struct
import time

import numpy as np
from PIL import Image

try:  # Pillow >=9.1
    from PIL import ImageDecompressionBombError
except ImportError:  # pragma: no cover - older Pillow fallback
    ImageDecompressionBombError = Image.DecompressionBombError  # type: ignore[attr-defined]

# Optional imports — guarded so the module works without them.
_ASTROPY_OK = False
_PHOTUTILS_OK = False
try:  # noqa: SIM105
    import astropy  # type: ignore
    from astropy.io import fits  # type: ignore
    from astropy.stats import sigma_clipped_stats  # type: ignore

    _ASTROPY_OK = True
except Exception:  # pragma: no cover - soft dependency
    fits = None  # type: ignore
    sigma_clipped_stats = None  # type: ignore

try:  # noqa: SIM105
    from photutils.aperture import (  # type: ignore
        CircularAperture,
        CircularAnnulus,
        aperture_photometry,
    )

    _PHOTUTILS_OK = True
except Exception:  # pragma: no cover - soft dependency
    CircularAperture = None  # type: ignore
    CircularAnnulus = None  # type: ignore
    aperture_photometry = None  # type: ignore


# --------------------------------------------------------------------------------------
# Image validation helpers
# --------------------------------------------------------------------------------------

class ImageValidationError(Exception):
    """Raised when input bytes are not a supported/safe image."""


@dataclass(frozen=True)
class ImageProbe:
    format: str
    mime: str
    width: Optional[int]
    height: Optional[int]
    extension: Optional[str] = None


_IMAGE_FORMATS = {
    "png": {
        "extensions": {".png"},
        "mime": "image/png",
    },
    "jpeg": {
        "extensions": {".jpg", ".jpeg"},
        "mime": "image/jpeg",
    },
    "gif": {
        "extensions": {".gif"},
        "mime": "image/gif",
    },
    "bmp": {
        "extensions": {".bmp"},
        "mime": "image/bmp",
    },
    "tiff": {
        "extensions": {".tif", ".tiff"},
        "mime": "image/tiff",
    },
    "webp": {
        "extensions": {".webp"},
        "mime": "image/webp",
    },
    "fits": {
        "extensions": {".fits", ".fit", ".fts"},
        "mime": "application/fits",
    },
}

_FORMAT_BY_EXTENSION = {
    ext: name for name, spec in _IMAGE_FORMATS.items() for ext in spec["extensions"]
}

_UNBOUNDED_FORMATS = {"fits"}  # formats where we'll defer full pixel limit checks

_DECODE_TIMEOUT_SECONDS = float(os.environ.get("ASTRO_IMAGE_DECODE_SECONDS", "3.0"))


def probe_image_bytes(data: bytes, filename: Optional[str] = None) -> ImageProbe:
    """Validate image bytes and return format/mime/dimensions without Pillow decode."""
    if not data:
        raise ImageValidationError("Uploaded file is empty.")

    extension = os.path.splitext(filename)[1].lower() if filename else None
    if extension and extension not in _FORMAT_BY_EXTENSION:
        raise ImageValidationError(
            f"Unsupported file extension: {extension}. Allowed: "
            + ", ".join(sorted(set(_FORMAT_BY_EXTENSION)))
        )

    # FITS is handled separately
    if extension in _IMAGE_FORMATS["fits"]["extensions"] or data.startswith(b"SIMPLE  ="):
        return _probe_fits(data, extension)

    # If extension known, try that format first
    candidate_formats = []
    if extension and extension in _FORMAT_BY_EXTENSION:
        candidate_formats.append(_FORMAT_BY_EXTENSION[extension])
    candidate_formats.extend(fmt for fmt in _IMAGE_FORMATS if fmt not in candidate_formats and fmt != "fits")

    expected_format = _FORMAT_BY_EXTENSION.get(extension) if extension else None
    for fmt in candidate_formats:
        try:
            width, height = _probe_dimensions(fmt, data)
            spec = _IMAGE_FORMATS[fmt]
            return ImageProbe(format=fmt, mime=spec["mime"], width=width, height=height, extension=extension)
        except ImageValidationError as exc:
            if expected_format == fmt:
                raise ImageValidationError(
                    f"File extension {extension} does not match the {fmt.upper()} signature: {exc}"
                ) from exc
            continue
        except Exception:
            continue

    raise ImageValidationError("File signature does not match supported image formats.")


def _probe_dimensions(fmt: str, data: bytes) -> Tuple[Optional[int], Optional[int]]:
    if fmt == "png":
        return _probe_png(data)
    if fmt == "jpeg":
        return _probe_jpeg(data)
    if fmt == "gif":
        return _probe_gif(data)
    if fmt == "bmp":
        return _probe_bmp(data)
    if fmt == "tiff":
        return _probe_tiff(data)
    if fmt == "webp":
        return _probe_webp(data)
    raise ImageValidationError(f"Unsupported format {fmt}")


def _ensure_positive_dimensions(width: Optional[int], height: Optional[int], fmt: str) -> Tuple[Optional[int], Optional[int]]:
    if width is not None and height is not None:
        if width <= 0 or height <= 0:
            raise ImageValidationError(f"{fmt.upper()} image has non-positive dimensions ({width}x{height}).")
    return width, height


def _probe_png(data: bytes) -> Tuple[int, int]:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageValidationError("Invalid PNG signature.")
    if data[12:16] != b"IHDR":
        raise ImageValidationError("PNG missing IHDR chunk.")
    width = int.from_bytes(data[16:20], "big", signed=False)
    height = int.from_bytes(data[20:24], "big", signed=False)
    return _ensure_positive_dimensions(width, height, "png")


def _probe_gif(data: bytes) -> Tuple[int, int]:
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ImageValidationError("Invalid GIF signature.")
    width = int.from_bytes(data[6:8], "little", signed=False)
    height = int.from_bytes(data[8:10], "little", signed=False)
    return _ensure_positive_dimensions(width, height, "gif")


def _probe_bmp(data: bytes) -> Tuple[int, int]:
    if len(data) < 26 or data[:2] != b"BM":
        raise ImageValidationError("Invalid BMP signature.")
    width = int.from_bytes(data[18:22], "little", signed=True)
    height = abs(int.from_bytes(data[22:26], "little", signed=True))
    return _ensure_positive_dimensions(width, height, "bmp")


def _probe_jpeg(data: bytes) -> Tuple[int, int]:
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise ImageValidationError("Invalid JPEG signature.")
    index = 2
    while index < len(data) - 1:
        if data[index] != 0xFF:
            break
        marker = data[index + 1]
        index += 2
        if marker in (0xD8, 0xD9, 0x01):
            continue
        if index + 2 > len(data):
            break
        segment_length = int.from_bytes(data[index:index + 2], "big", signed=False)
        if segment_length < 2 or index + segment_length > len(data):
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if segment_length < 7:
                break
            height = int.from_bytes(data[index + 3:index + 5], "big", signed=False)
            width = int.from_bytes(data[index + 5:index + 7], "big", signed=False)
            return _ensure_positive_dimensions(width, height, "jpeg")
        index += segment_length
    raise ImageValidationError("JPEG dimensions not found in header.")


def _probe_tiff(data: bytes) -> Tuple[int, int]:
    if len(data) < 8:
        raise ImageValidationError("TIFF header too short.")
    byte_order = data[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        raise ImageValidationError("Invalid TIFF byte order.")
    if struct.unpack(endian + "H", data[2:4])[0] != 42:
        raise ImageValidationError("Invalid TIFF magic number.")
    offset = struct.unpack(endian + "I", data[4:8])[0]
    if offset + 2 > len(data):
        raise ImageValidationError("TIFF IFD offset outside file.")
    entries = struct.unpack(endian + "H", data[offset:offset + 2])[0]
    ptr = offset + 2
    width = height = None
    for _ in range(entries):
        if ptr + 12 > len(data):
            break
        tag = struct.unpack(endian + "H", data[ptr:ptr + 2])[0]
        typ = struct.unpack(endian + "H", data[ptr + 2:ptr + 4])[0]
        count = struct.unpack(endian + "I", data[ptr + 4:ptr + 8])[0]
        value_offset = data[ptr + 8:ptr + 12]
        if tag in (256, 257):  # ImageWidth / ImageLength
            if typ == 3:  # SHORT
                value = struct.unpack(endian + "H", value_offset[:2])[0]
            elif typ == 4:  # LONG
                value = struct.unpack(endian + "I", value_offset)[0]
            else:
                raise ImageValidationError("Unsupported TIFF type for dimensions.")
            if tag == 256:
                width = int(value)
            else:
                height = int(value)
        ptr += 12
        if width is not None and height is not None:
            break
    if width is None or height is None:
        raise ImageValidationError("TIFF dimensions not found.")
    return _ensure_positive_dimensions(width, height, "tiff")


def _probe_webp(data: bytes) -> Tuple[int, int]:
    if len(data) < 16 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ImageValidationError("Invalid WebP signature.")
    chunk_type = data[12:16]
    if chunk_type == b"VP8X":
        if len(data) < 30:
            raise ImageValidationError("WebP VP8X header too short.")
        width_minus_one = int.from_bytes(data[24:27], "little", signed=False)
        height_minus_one = int.from_bytes(data[27:30], "little", signed=False)
        width = width_minus_one + 1
        height = height_minus_one + 1
    elif chunk_type == b"VP8 ":
        data_start = 20
        if len(data) < data_start + 10:
            raise ImageValidationError("WebP VP8 header too short.")
        width = int.from_bytes(data[data_start + 6:data_start + 8], "little", signed=False)
        height = int.from_bytes(data[data_start + 8:data_start + 10], "little", signed=False)
    elif chunk_type == b"VP8L":
        if len(data) < 21:
            raise ImageValidationError("WebP VP8L header too short.")
        # VP8L stores 14-bit width/height as described in spec
        b0, b1, b2, b3, b4 = data[20:25]
        width = ((b1 & 0x3F) << 8 | b0) + 1
        height = ((b3 & 0x0F) << 10 | (b2 << 2) | (b1 >> 6)) + 1
    else:
        raise ImageValidationError("Unsupported WebP chunk type.")
    return _ensure_positive_dimensions(width, height, "webp")


def _probe_fits(data: bytes, extension: Optional[str]) -> ImageProbe:
    if not data.startswith(b"SIMPLE  ="):
        raise ImageValidationError("Invalid FITS signature.")
    if not _ASTROPY_OK or fits is None:
        raise ImageValidationError("FITS support requires astropy to be installed.")
    header_block = data[:2880]
    if len(header_block) < 80:
        raise ImageValidationError("FITS header too short.")
    cards = [header_block[i:i + 80] for i in range(0, len(header_block), 80)]
    naxis = {}
    for card in cards:
        key = card[:8].decode("ascii", errors="ignore").strip()
        value = card[10:30].strip()
        if key.startswith("NAXIS"):
            try:
                naxis[int(key[5:])] = int(value)
            except Exception:
                continue
        if key == "END":
            break
    width = naxis.get(1)
    height = naxis.get(2)
    spec = _IMAGE_FORMATS["fits"]
    return ImageProbe(format="fits", mime=spec["mime"], width=width, height=height, extension=extension)


# --------------------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------------------

def has_real_photometry() -> bool:
    """Return True if astropy + photutils are importable.

    We require **both** for full aperture photometry support.
    """
    return bool(_ASTROPY_OK and _PHOTUTILS_OK)


def simple_brightness(image: Union[str, bytes, Image.Image, np.ndarray]) -> float:
    """Return a lightweight brightness estimate in [0..1] (approx.).

    We convert to grayscale by averaging channels, then take the mean of all
    pixels. This is meant for quick heuristics when scientific photometry
    deps aren't installed.
    """
    arr = _to_float_array(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)  # average channels
    # Normalize to [0, 1] if input looked like 8/16-bit; otherwise assume floats.
    arr = _auto_normalize(arr)
    return float(np.clip(arr.mean(), 0.0, 1.0))


def measure_brightness(
    image: Union[str, bytes, Image.Image, np.ndarray],
    positions: Optional[Sequence[Tuple[float, float]]] = None,
    r: Optional[float] = None,
    r_in: Optional[float] = None,
    r_out: Optional[float] = None,
) -> Union[float, List[dict]]:
    """Measure brightness for an image.

    - If real photometry libs are present **and** `positions` and `r` are given,
      perform circular-aperture photometry with optional local background
      subtraction via an annulus defined by `r_in`, `r_out`.
      Returns a list of dicts (one per position) with keys:
        {'x','y','r','aperture_sum','bkg_mean','bkg_area','flux_sub'}

    - Otherwise, return a single float from `simple_brightness(image)`.
    """
    if has_real_photometry() and r is not None:
        # ✅ Правка 2: если positions пуст/None — вернуть пустой список
        # Explicitly handle None or empty sequences/arrays.
        # Avoid `if not positions` because a numpy array has truthiness ambiguity.
        if positions is None or len(positions) == 0:
            return []
        return _aperture_photometry(image, positions, r, r_in=r_in, r_out=r_out)
    return simple_brightness(image)


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

ImgLike = Union[str, bytes, Image.Image, np.ndarray]


def _auto_normalize(arr: np.ndarray) -> np.ndarray:
    """Best-effort normalization.

    If the array looks like 8- or 16-bit integers, scale to 0..1.
    Otherwise, return as-is.
    """
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.max > 0:
            return arr.astype(np.float64) / float(info.max)
        return arr.astype(np.float64)
    # float images: attempt robust scaling if values look large
    if arr.dtype.kind == 'f':
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            return arr
        if vmax <= 1.0 and vmin >= 0.0:
            return arr
        if vmax <= 255.0 and vmin >= 0.0:
            return (arr / 255.0).astype(np.float64)
    return arr


def _to_float_array(
    image: ImgLike,
    *,
    filename: Optional[str] = None,
    probe: Optional[ImageProbe] = None,
) -> np.ndarray:
    """Convert supported inputs into a float array (H, W) or (H, W, C).

    - Str path: load via Pillow; if `.fits` and astropy exists, load via astropy.
    - Bytes: decode via Pillow.
    - PIL.Image: convert to RGB(A) if needed.
    - np.ndarray: pass-through.
    """
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float64, copy=False)
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float64)
    elif isinstance(image, (str, os.PathLike)):
        path = str(image)
        with open(path, "rb") as fh:
            data = fh.read()
        if probe is None:
            probe = probe_image_bytes(data, filename=path)
        arr = _to_float_array(data, filename=path, probe=probe)
    elif isinstance(image, (bytes, bytearray, memoryview)):
        data_bytes = bytes(image)
        if probe is None:
            probe = probe_image_bytes(data_bytes, filename=filename)
        if probe.format == "fits":
            if not _ASTROPY_OK or fits is None:
                raise ImageValidationError("FITS support requires astropy to be installed.")
            with fits.open(io.BytesIO(data_bytes), memmap=False) as hdul:
                hdu = hdul[0]
                if hdu.data is None:
                    raise ImageValidationError("FITS file has no primary image data.")
                arr = np.array(hdu.data, dtype=np.float64, copy=False)
        else:
            decode_start = time.perf_counter()
            try:
                with Image.open(io.BytesIO(data_bytes)) as im:
                    arr = np.array(im.convert("RGB"), dtype=np.float64)
            except ImageDecompressionBombError as exc:
                raise ImageValidationError("Image is too large to process safely.") from exc
            except Exception as exc:
                raise ImageValidationError(f"Failed to decode image: {exc}") from exc
            elapsed = time.perf_counter() - decode_start
            if elapsed > _DECODE_TIMEOUT_SECONDS:
                raise ImageValidationError(
                    f"Image decoding exceeded {_DECODE_TIMEOUT_SECONDS:.1f}s safety limit."
                )
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    return arr


@dataclass
class _PhotometryConfig:
    r: float
    r_in: Optional[float]
    r_out: Optional[float]


def _aperture_photometry(
    image: ImgLike,
    positions: Sequence[Tuple[float, float]],
    r: float,
    r_in: Optional[float] = None,
    r_out: Optional[float] = None,
) -> List[dict]:
    if not has_real_photometry():  # defensive guard
        raise RuntimeError("Aperture photometry requested but photutils/astropy not available.")

    arr = _to_float_array(image)
    # If RGB, convert to luminance-like grayscale for photometry (simple mean).
    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    # ✅ Правка 1: НЕ нормализуем данные перед апертурной фотометрией — сохраняем счётчики
    data = arr.astype(np.float64, copy=False)

    # Build apertures / annuli
    aper = CircularAperture(positions, r=r)
    annulus = None
    if r_in is not None and r_out is not None and r_out > r_in > 0:
        annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    # Perform raw aperture photometry
    phot_table = aperture_photometry(data, aper)

    # Background estimation per position (if annulus provided)
    bkg_means: List[float] = [0.0] * len(positions)
    bkg_areas: List[float] = [0.0] * len(positions)

    if annulus is not None:
        # Sample annulus pixels to compute sigma-clipped stats per aperture
        annulus_masks = annulus.to_mask(method="center")
        for i, m in enumerate(annulus_masks):
            annulus_data = m.multiply(data)  # shape ≈ local cutout; zeros/NANs outside mask
            # ✅ Правка 3: защита — берём только конечные и действительно замаскированные пиксели
            valid = np.isfinite(annulus_data) & (m.data > 0)
            if not np.any(valid):
                bkg_means[i] = 0.0
                bkg_areas[i] = 0.0
                continue

            vals = annulus_data[valid]
            # если очень мало точек — считаем это недостаточно значимо
            if vals.size < 4:
                bkg_means[i] = 0.0
                bkg_areas[i] = float(valid.sum())
                continue

            if sigma_clipped_stats is not None:
                _, med, _ = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
                bkg_means[i] = float(med)
            else:
                bkg_means[i] = float(np.median(vals))
            # Площадь фоновой выборки — количество валидных пикселей (в пикселях)
            bkg_areas[i] = float(valid.sum())

    # Compose results
    results: List[dict] = []
    ap_area = float(np.pi * (float(r) ** 2))
    for idx, (x, y) in enumerate(positions):
        ap_sum = float(phot_table["aperture_sum"][idx])
        bmean = bkg_means[idx] if idx < len(bkg_means) else 0.0
        # Background-subtracted flux using среднее по фону * площадь апертуры
        flux_sub = ap_sum - (bmean * ap_area)
        results.append(
            {
                "x": float(x),
                "y": float(y),
                "r": float(r),
                "aperture_sum": ap_sum,
                "bkg_mean": float(bmean),
                "bkg_area": float(bkg_areas[idx]) if idx < len(bkg_areas) else 0.0,
                "flux_sub": float(flux_sub),
            }
        )

    return results
