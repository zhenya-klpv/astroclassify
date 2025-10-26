from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PSFPhotometryResult",
    "PSFModel",
    "run_psf_photometry",
]


@dataclass(frozen=True)
class PSFModel:
    """Container describing the averaged ePSF kernel."""

    kernel: np.ndarray
    center: Tuple[float, float]
    fwhm_major: float
    fwhm_minor: float
    ellipticity: float
    position_angle: float


@dataclass(frozen=True)
class PSFPhotometryResult:
    """Per-source PSF-fit output."""

    flux_psf: float
    flux_err_psf: float
    snr_psf: float
    chi2_psf: float
    fwhm_major: float
    fwhm_minor: float
    ellipticity: float
    position_angle: float
    background: float


def _extract_stamp(
    image: np.ndarray,
    x: float,
    y: float,
    radius: int,
) -> Optional[np.ndarray]:
    x0 = int(round(x))
    y0 = int(round(y))
    size = 2 * radius + 1
    x_start = x0 - radius
    y_start = y0 - radius
    x_end = x_start + size
    y_end = y_start + size
    if x_start < 0 or y_start < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
        return None
    return image[y_start:y_end, x_start:x_end].astype(np.float64, copy=False)


def _estimate_background(stamp: np.ndarray, mask_radius: int) -> Tuple[float, float]:
    yy, xx = np.indices(stamp.shape)
    cx = (stamp.shape[1] - 1) / 2.0
    cy = (stamp.shape[0] - 1) / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    annulus = stamp[rr >= mask_radius]
    if annulus.size == 0:
        annulus = stamp
    background = float(np.median(annulus))
    sigma = float(np.std(annulus)) if annulus.size > 1 else math.sqrt(abs(background))
    return background, sigma


def _build_epsf(stamps: List[np.ndarray]) -> Optional[PSFModel]:
    if not stamps:
        return None
    accum = None
    weights = 0.0
    for stamp in stamps:
        background, _ = _estimate_background(stamp, mask_radius=max(stamp.shape) // 2 - 1)
        signal = stamp - background
        total = float(signal.sum())
        if total <= 0:
            continue
        normalized = signal / total
        if accum is None:
            accum = normalized
        else:
            accum += normalized
        weights += 1.0
    if accum is None or weights == 0:
        return None
    epsf = accum / weights

    yy, xx = np.indices(epsf.shape)
    total = float(epsf.sum())
    if total <= 0:
        return None
    cx = float((epsf * xx).sum() / total)
    cy = float((epsf * yy).sum() / total)
    dx = xx - cx
    dy = yy - cy
    cov_xx = float((epsf * dx * dx).sum() / total)
    cov_yy = float((epsf * dy * dy).sum() / total)
    cov_xy = float((epsf * dx * dy).sum() / total)
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, a_min=1e-12, a_max=None)
    idx_major = int(np.argmax(evals))
    idx_minor = 1 - idx_major
    sigma_major = math.sqrt(float(evals[idx_major]))
    sigma_minor = math.sqrt(float(evals[idx_minor]))
    fwhm_major = 2.354820045 * sigma_major
    fwhm_minor = 2.354820045 * sigma_minor
    ellipticity = 1.0 - (sigma_minor / sigma_major if sigma_major > 0 else 0.0)
    vec = evecs[:, idx_major]
    position_angle = math.degrees(math.atan2(vec[1], vec[0]))
    if position_angle < 0:
        position_angle += 180.0

    return PSFModel(
        kernel=epsf,
        center=(cx, cy),
        fwhm_major=fwhm_major,
        fwhm_minor=fwhm_minor,
        ellipticity=ellipticity,
        position_angle=position_angle,
    )


def run_psf_photometry(
    image: np.ndarray,
    positions: Sequence[Tuple[float, float]],
    *,
    stamp_radius: int = 6,
    fit_radius: int = 4,
) -> Tuple[List[Tuple[int, PSFPhotometryResult]], Optional[PSFModel]]:
    """Simple ePSF builder + matched-filter photometry."""
    if len(positions) == 0:
        return [], None
    stamps: List[np.ndarray] = []
    valid_entries: List[Tuple[int, Tuple[float, float], np.ndarray]] = []
    for idx, (x, y) in enumerate(positions):
        stamp = _extract_stamp(image, x, y, stamp_radius)
        if stamp is None:
            continue
        stamps.append(stamp)
        valid_entries.append((idx, (x, y), stamp))
    epsf_model = _build_epsf(stamps)
    if epsf_model is None:
        return [], None

    kernel = epsf_model.kernel
    flat_kernel = kernel.flatten()
    norm = float(np.dot(flat_kernel, flat_kernel))
    if norm <= 0:
        return [], epsf_model

    results: List[Tuple[int, PSFPhotometryResult]] = []
    for (idx, (x, y), stamp) in valid_entries:
        background, sigma_bg = _estimate_background(stamp, mask_radius=fit_radius + 2)
        signal = stamp - background
        flat_signal = signal.flatten()
        flux = float(np.dot(flat_kernel, flat_signal) / norm)
        residual = flat_signal - flux * flat_kernel
        chi2 = float(np.mean(residual**2))

        n_pix = float((kernel > 0).sum())
        gain_equivalent = 1.0
        variance = max(flux * gain_equivalent, 0.0) + (sigma_bg**2 * n_pix)
        flux_err = math.sqrt(max(variance, 1.0))
        snr = flux / flux_err if flux_err > 0 else 0.0

        results.append(
            (
                idx,
                PSFPhotometryResult(
                    flux_psf=float(flux),
                    flux_err_psf=float(flux_err),
                    snr_psf=float(snr),
                    chi2_psf=float(chi2),
                    fwhm_major=epsf_model.fwhm_major,
                    fwhm_minor=epsf_model.fwhm_minor,
                    ellipticity=epsf_model.ellipticity,
                    position_angle=epsf_model.position_angle,
                    background=float(background),
                ),
            )
        )
    return results, epsf_model
