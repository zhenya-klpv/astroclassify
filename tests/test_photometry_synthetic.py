from __future__ import annotations

import math
from pathlib import Path
import importlib.util

import numpy as np
import pytest

try:  # optional dependency in tests
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - fastapi not installed
    TestClient = None  # type: ignore

from astroclassify.api.io_export import export_photometry
try:  # FastAPI optional in minimal envs
    from astroclassify.api.main import app
except Exception:  # pragma: no cover - FastAPI not installed
    app = None  # type: ignore
from astroclassify.api.photometry import ImageValidationError, measure_brightness, probe_image_bytes, simple_brightness
from astroclassify.psf_photometry import run_psf_photometry


def test_simple_brightness_gaussian(gaussian_frame: callable) -> None:
    frame = gaussian_frame(
        size=96,
        centers=[(32.0, 38.0), (60.0, 54.0)],
        fwhm=[3.2, 5.5],
        amplitudes=[4000.0, 2500.0],
        background=12.0,
        noise=0.0,
    )
    brightness = simple_brightness(frame)
    assert brightness == pytest.approx(1.0, rel=1e-6)


def test_probe_image_bytes_rejects_unknown_extension(png_bytes_from_array: callable) -> None:
    array = np.ones((32, 32), dtype=np.float32) * 42.0
    data = png_bytes_from_array(array)
    with pytest.raises(ImageValidationError):
        probe_image_bytes(data, filename="frame.xyz")


def test_export_photometry_csv_matches_golden(golden_dir: Path) -> None:
    results = [
        {
            "x": 12.5,
            "y": 18.75,
            "r": 4.0,
            "aperture_sum": 1234.5678,
            "bkg_mean": 12.0,
            "bkg_area": 50.265482,
            "flux_sub": 1234.5678 - 12.0 * math.pi * 4.0**2,
        }
    ]
    artifact = export_photometry(results, "csv")
    expected_path = golden_dir / "export_photometry_psf.csv"
    actual = artifact.content.decode("utf-8").strip().replace("\r\n", "\n")
    expected = expected_path.read_text().strip().replace("\r\n", "\n")
    assert actual == expected


PHOTOMETRY_DEPS_PRESENT = (
    importlib.util.find_spec("photutils") is not None
    and importlib.util.find_spec("astropy") is not None
)


@pytest.mark.skipif(not PHOTOMETRY_DEPS_PRESENT, reason="photometry dependencies not installed")
def test_measure_brightness_matches_expected(gaussian_frame: callable) -> None:
    frame = gaussian_frame(
        size=96,
        centers=[(40.0, 45.0)],
        fwhm=[4.0],
        amplitudes=[5000.0],
        background=8.0,
        noise=0.0,
    )
    positions = [(40.0, 45.0)]
    result = measure_brightness(frame, positions=positions, r=6.0, r_in=8.0, r_out=12.0)
    assert isinstance(result, list) and len(result) == 1
    flux = float(result[0]["flux_sub"])
    sigma = 4.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    expected_flux = 5000.0 * 2.0 * math.pi * sigma**2 * (1.0 - math.exp(-(6.0**2) / (2.0 * sigma**2)))
    assert flux == pytest.approx(expected_flux, rel=0.05)


def test_psf_flux_handles_blending(gaussian_frame: callable) -> None:
    frame = gaussian_frame(
        size=80,
        centers=[(36.5, 40.2), (42.0, 43.0)],
        fwhm=[3.0, 3.5],
        amplitudes=[8000.0, 6000.0],
        background=15.0,
        noise=0.0,
    )
    positions = [(36.5, 40.2), (42.0, 43.0)]
    psf_entries, psf_model = run_psf_photometry(frame, positions, stamp_radius=6, fit_radius=4)
    assert psf_model is not None and psf_entries

    def expected_flux(amp: float, fwhm: float) -> float:
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return amp * 2.0 * math.pi * sigma**2

    expected_values = [expected_flux(8000.0, 3.0), expected_flux(6000.0, 3.5)]

    for (idx, result), expected in zip(psf_entries, expected_values):
        psf_flux = result.flux_psf
        assert psf_flux == pytest.approx(expected, rel=0.6)


@pytest.mark.skipif(not PHOTOMETRY_DEPS_PRESENT, reason="photometry dependencies not installed")
def test_psf_magnitude_calibration(gaussian_frame: callable, png_bytes_from_array: callable) -> None:
    if TestClient is None or app is None:
        pytest.skip("fastapi test client not available")
    client = TestClient(app)
    frame = gaussian_frame(
        size=96,
        centers=[(40.0, 45.0)],
        fwhm=[4.0],
        amplitudes=[5000.0],
        background=8.0,
        noise=0.0,
    )
    image_bytes = png_bytes_from_array(frame)
    response = client.post(
        "/v1/detect_sources?xy=40,45&r=6&phot_mode=psf&zeropoint=25&exptime=30&gain=1",
        files={"file": ("psf.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("photometry_mode") == "psf"
    result = payload["results"][0]
    assert "mag" in result and "mag_err" in result
    sigma = 4.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    expected_flux = 5000.0 * 2.0 * math.pi * sigma**2
    counts_per_sec = expected_flux / 30.0
    expected_mag = 25.0 - 2.5 * math.log10(counts_per_sec)
    assert result["mag"] == pytest.approx(expected_mag, rel=0.15)
