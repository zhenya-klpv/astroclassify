# 🌌 AstroClassify — Photometry & Source Detection API

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-🚀-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-success)](#)
[![Prometheus Ready](https://img.shields.io/badge/metrics-prometheus-blue.svg)](#)

---

**AstroClassify** is a lightweight, secure, and modular service for **astrophotometry** and **source detection** — built with FastAPI, Prometheus, and Photutils.  
It can ingest FITS/PNG/JPEG frames, detect stellar sources, measure brightness, and export structured results (JSON/CSV/FITS).

---

## ✨ Features

- **Safe file handling** — upload limits, MIME validation, decompression-bomb protection (`Pillow.MAX_IMAGE_PIXELS`)
- **Thread-safe inference** — semaphore-controlled concurrent operations via `ASTRO_MAX_CONCURRENT_INFERENCES`
- **RFC5987 filename sanitization** + JSON-safe numeric outputs (no `NaN` / `Inf`)
- **Prometheus metrics** — `/metrics` endpoint, multiprocess-ready for Gunicorn
- **Modular architecture**
  - `astroclassify/api` — FastAPI endpoints  
  - `astroclassify/core` — device & concurrency utilities  
  - `astroclassify/api/photometry.py` — real & fallback photometry modes
- **Photometry endpoints**
- `/v1/detect_sources` — manual aperture / simple brightness  
- `/v1/detect_auto` — DAO / SEP auto-detection + photometry (export: `format=json|csv|fits`, `bundle=zip`, morphology columns FWHM/ellipticity/PA)  
- `/v1/preview_apertures` — preview overlays with diagnostics (overlay/panel layouts, PNG or ZIP bundle)
- **PSF mode** — `phot_mode=psf` builds an ePSF kernel per request, outputs PSF flux, FWHM, ellipticity, SNR, and (when calibrated) `mag`/`mag_err`
- **Developer tooling** — multi-stage Dockerfiles, Grafana dashboard, and a first-party `astrocli` helper for scripted workflows
- **Versioned API** — all routes live under `/v1/...` and responses expose `X-AstroClassify-API: 1`
- **Smoke test suite (8 tests)** — ensures API stability and response consistency  
- **Prometheus-compatible metrics** — `ac_http_requests_total`, latency histograms, etc.


## 🚀 Quickstart (≤5 minutes)

### Option A — Python virtualenv

```bash
git clone https://github.com/zhenya-klpv/astroclassify.git
cd astroclassify
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn astroclassify.api.main:app --host 0.0.0.0 --port 8000
```

Browse to http://127.0.0.1:8000/docs for the interactive OpenAPI UI.

### Option B — Docker Compose

```bash
docker compose up --build api-cpu
# GPU profile (requires NVIDIA Container Toolkit)
docker compose --profile gpu up --build api-gpu
```

Both images ship with a `/v1/health` healthcheck. Logs expose the listening URL and metrics directory.

### Option C — astrocli helper

The repository now ships with a first-party CLI. Once the virtualenv is active (or the package is installed), run:

```bash
astrocli detect --file tiny.fits --detector sep --threshold 2.5 --format json
astrocli preview --file tiny.fits.preview.png --xy 120,80 --layout panel --plots radial,growth --output preview.png
astrocli detect --file tiny.fits --detector sep --phot-mode psf --zeropoint 25 --exptime 180 --gain 1.2 --format json
```

Pass `--dry-run` to inspect the request instead of executing it.

---

## 🔭 Ready-to-use curl recipes

Narrow PSF (well-sampled stars, low background):

```bash
curl -F "file=@tiny.fits" \
  "http://127.0.0.1:8000/v1/detect_auto?detector=sep&threshold_sigma=1.8&max_sources=200&r=4&r_in=6&r_out=9"
```

Wide PSF (seeing-limited data, fat FWHM):

```bash
curl -F "file=@tiny.fits" \
  "http://127.0.0.1:8000/v1/detect_auto?detector=dao&threshold_sigma=3.5&max_sources=150&r=8&r_in=11&r_out=16&r_mode=auto&r_factor=1.8"
```

High background (nebulae / crowded light pollution):

```bash
curl -F "file=@tiny.fits" \
  "http://127.0.0.1:8000/v1/detect_auto?detector=sep&threshold_sigma=4.5&bkg=global&sep_minarea=9&sep_filter_kernel=gaussian"
```

Dense fields (globular clusters — keep more detections, export JSON + preview bundle):

```bash
curl -o cluster.bundle.zip -F "file=@tiny.fits" \
  "http://127.0.0.1:8000/v1/detect_auto?detector=sep&threshold_sigma=2.2&max_sources=500&format=json&bundle=zip&download=true"
```

PSF photometry with calibrated magnitudes:

```bash
curl -F "file=@tiny.fits" \
  "http://127.0.0.1:8000/v1/detect_sources?xy=120,80&r=5&phot_mode=psf&zeropoint=25.3&exptime=90&gain=1.4&mag_system=AB"
```

Manual apertures with labelled preview:

```bash
curl -o preview.png -F "file=@tiny.fits.preview.png" \
  "http://127.0.0.1:8000/v1/preview_apertures?xy=120,80&xy=200,140&r=6&r_in=10&r_out=15&layout=panel&plots=radial,growth"
```

Export calibrated photometry (CSV):

```bash
curl -o photometry.csv -F "file=@tiny.fits" \
  "http://127.0.0.1:8000/v1/detect_sources?xy=120,80&r=5&format=csv&download=true"
```

Remote survey cutouts:

```bash
# NOIRLab NSC / DECam
curl -o noirlab.fits \
  "http://127.0.0.1:8000/v1/cutout?service=noirlab&ra=210.8024&dec=54.3487&size_arcmin=2&band=i"

# Pan-STARRS via MAST (requires astroquery)
curl -o panstarrs.fits \
  "http://127.0.0.1:8000/v1/cutout?service=mast&ra=210.8024&dec=54.3487&size_arcmin=1.5&filters=g,r,i"

# SDSS DR17 JPEG preview
curl -o sdss.jpg \
  "http://127.0.0.1:8000/v1/cutout?service=sdss&ra=210.8024&dec=54.3487&size_arcsec=90"
```

---

## ⚙️ Detector & aperture parameters

| Parameter | Applies to | Default | Purpose |
|-----------|------------|---------|---------|
| `threshold_sigma` | SEP / DAO | `3.0` | Detection threshold in σ units |
| `max_sources` | SEP / DAO | `50` | Limit number of returned sources |
| `r`, `r_in`, `r_out` | All | `5 / 8 / 12` | Aperture radius and background annulus (px) |
| `r_mode`, `r_factor` | DAO | `manual / 2.0` | Auto-radius selection based on FWHM |
| `sep_minarea` | SEP | `5` | Minimum connected pixels over threshold |
| `sep_filter_kernel` | SEP | `3x3` | Smoothing kernel (`3x3`, `gaussian`, `none`) |
| `sep_deblend_nthresh` | SEP | `32` | Number of deblend thresholds |
| `sep_deblend_cont` | SEP | `0.005` | Deblend contrast ratio |
| `dao_sharplo`, `dao_roundlo`, `dao_roundhi` | DAO | `0.2 / -1.0 / 1.0` | Shape filters for DAOStarFinder |
| `phot_mode` | Aperture/PSF | `aperture` | Set to `psf` to enable ePSF photometry and magnitudes |
| `psf_stamp`, `psf_fit` | PSF | `8 / 4` | Stamp radius and fit radius for PSF model |
| `zeropoint` | Calibration | `None` | Photometric zero point (adds `mag`/`mag_err`) |
| `exptime` | Calibration | `None` | Exposure time in seconds (used for mag calculation) |
| `gain` | Calibration | `None` | Detector gain in e-/ADU |
| `mag_system` | Calibration | `AB` | Label stored with exported magnitudes |

Combine these with the recipes above or pass them to `astrocli` / `curl` as query parameters.

📊 Metrics

Prometheus-compatible counters and histograms are exposed at `/v1/metrics` (and `/metrics` for legacy clients). Grafana dashboard JSON lives in `ops/grafana/astroclassify_dashboard.json` and ships with panels for RPS, error codes, latency quantiles, source counts, and slow endpoints.

🩺 Health & Readiness

- `/v1/health` — fast ping used by load balancers
- `/v1/ready` — dependency probe (photutils/sep availability), temp-directory write test, ResNet model status, torch/torchvision versions

Both endpoints return JSON payloads with `status` and per-check flags so Kubernetes, Docker, or systemd can gate traffic accurately.

🧪 Tests

Regression tests now cover synthetic PSF frames, photometry exports, CLI dry-runs, and API smoke requests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids ROS/colcon plugins interfering with collection.)

✨ PyTorch wheels

`requirements.txt` ships with CPU wheels (`torch==2.4.1`, `torchvision==0.19.1`). Add CUDA wheels after installation when running on NVIDIA hosts:

```bash
pip install --upgrade torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

If PyTorch is absent, `/v1/classify*` responds with `ASTRO_5022` while photometry and preview APIs remain fully functional.

🧱 Architecture Overview
``` bash
astroclassify/
├── api/
│   ├── main.py            ← FastAPI app, routes, Prometheus registry, readiness checks
│   └── photometry.py      ← aperture & fallback photometry + input validation
├── cli.py                 ← `astrocli` entrypoint (detect / preview helpers)
├── core/device.py         ← device selection + concurrency limits
├── tests/                 ← smoke tests, synthetic PSF fixtures, CLI coverage
└── ops/
    ├── docker/            ← CPU & GPU Dockerfiles
    └── grafana/           ← Importable Grafana dashboard JSON
```

🧩 Roadmap
Milestone	Description
✅ v0.9.0	Prometheus + Photometry core
⏳ v1.0.0	CLI, Docker, GitHub Actions CI, documentation polish
💡 v1.1.0	Web dashboard + async endpoints

📦 Releases & Versioning

AstroClassify follows [semantic versioning](https://semver.org/). Container tags, PyPI releases, and SBOM artefacts use the same `MAJOR.MINOR.PATCH` cadence. The GitHub release checklist includes:

1. `pre-commit run --all-files`
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`
3. `pre-commit run cyclonedx-sbom --all-files`
4. Update `CHANGELOG.md` (when present) and bump the version in `pyproject.toml`
5. Tag the release (`git tag vX.Y.Z && git push --tags`)

🧰 For Developers

```bash
pip install -r requirements-dev.in
pre-commit install
pre-commit run --all-files
# produce an SBOM (CycloneDX JSON)
pre-commit run cyclonedx-sbom --all-files
```

The default hook suite runs **black**, **ruff**, **isort**, **mypy**, **reuse** (license headers), and produces a CycloneDX SBOM (`sbom.json`).

Hot-reload during development:

```bash
uvicorn astroclassify.api.main:app --reload
```
📄 License

This project is released under the MIT License.

Author: @zhenya-klpv
💫 AstroClassify — Open, modular, and observatory-grade photometry engine.
