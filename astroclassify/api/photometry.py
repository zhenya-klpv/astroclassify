"""Photometry utilities for AstroClassify.

Two modes of operation:

1) **Real aperture photometry** (preferred):
   If `photutils` and `astropy` are available, we do proper circular-aperture
   photometry with optional local background subtraction via a circular annulus.

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

import numpy as np
from PIL import Image

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
    if has_real_photometry() and positions is not None and r is not None:
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
        # If values span a very small range around zero, keep as-is.
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
            return arr
        # Heuristic: if within a typical 0..1 or 0..255 range, normalize accordingly.
        if vmax <= 1.0 and vmin >= 0.0:
            return arr
        if vmax <= 255.0 and vmin >= 0.0:
            return (arr / 255.0).astype(np.float64)
    return arr


def _to_float_array(image: ImgLike) -> np.ndarray:
    """Convert supported inputs into a float array (H, W) or (H, W, C).

    - Str path: load via Pillow; if `.fits` and astropy exists, load via astropy.
    - Bytes: decode via Pillow.
    - PIL.Image: convert to RGB(A) if needed.
    - np.ndarray: pass-through.
    """
    if isinstance(image, np.ndarray):
        arr = image
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float64)
    elif isinstance(image, (str, os.PathLike)):
        p = str(image)
        ext = os.path.splitext(p)[1].lower()
        if ext in {".fits", ".fit", ".fts"}:
            if not _ASTROPY_OK or fits is None:
                raise RuntimeError("FITS input requires astropy to be installed.")
            with fits.open(p, memmap=True) as hdul:
                data = hdul[0].data
            if data is None:
                raise ValueError("FITS file has no primary image data.")
            arr = np.array(data, dtype=np.float64)
        else:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB"), dtype=np.float64)
    elif isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as im:
            arr = np.array(im.convert("RGB"), dtype=np.float64)
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

    data = _auto_normalize(arr).astype(np.float64, copy=False)

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
            annulus_data = m.multiply(data)
            # mask out zeros
            mask = m.data.astype(bool)
            vals = annulus_data[mask]
            if vals.size == 0:
                bkg_means[i] = 0.0
                bkg_areas[i] = 0.0
                continue
            if sigma_clipped_stats is not None:
                _, med, _ = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
                bkg_means[i] = float(med)
            else:
                bkg_means[i] = float(np.median(vals))
            bkg_areas[i] = float(mask.sum())

    # Compose results
    results: List[dict] = []
    areas = np.pi * (float(r) ** 2)
    for idx, (x, y) in enumerate(positions):
        ap_sum = float(phot_table["aperture_sum"][idx])
        bmean = bkg_means[idx] if idx < len(bkg_means) else 0.0
        # If we didn't compute area via mask (annulus None), areas[idx] stays analytic for aperture
        flux_sub = ap_sum - (bmean * areas)
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


# --------------------------------------------------------------------------------------
# Minimal CLI (dev convenience)
# --------------------------------------------------------------------------------------

def _parse_cli(argv: Sequence[str]) -> dict:
    import argparse

    p = argparse.ArgumentParser(description="AstroClassify photometry test tool")
    p.add_argument("image", type=str, help="Path to the image (PNG/JPG/TIFF/FITS)")
    p.add_argument("--x", type=float, default=None, help="Aperture center x")
    p.add_argument("--y", type=float, default=None, help="Aperture center y")
    p.add_argument("--r", type=float, default=None, help="Aperture radius (pixels)")
    p.add_argument("--r-in", type=float, default=None, help="Annulus inner radius (pixels)")
    p.add_argument("--r-out", type=float, default=None, help="Annulus outer radius (pixels)")

    ns = p.parse_args(argv)
    return {
        "image": ns.image,
        "x": ns.x,
        "y": ns.y,
        "r": ns.r,
        "r_in": ns.r_in,
        "r_out": ns.r_out,
    }


def _run_cli(args: dict) -> None:
    img = args["image"]
    x, y, r = args["x"], args["y"], args["r"]
    r_in, r_out = args["r_in"], args["r_out"]

    if x is not None and y is not None and r is not None and has_real_photometry():
        res = measure_brightness(img, positions=[(x, y)], r=r, r_in=r_in, r_out=r_out)
        print(res)
    else:
        val = simple_brightness(img)
        print({"simple_brightness": val, "real_photometry_available": has_real_photometry()})


if __name__ == "__main__":
    import sys

    _run_cli(_parse_cli(sys.argv[1:]))
