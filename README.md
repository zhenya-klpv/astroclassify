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
  - `/detect_sources` — manual aperture / simple brightness  
  - `/detect_auto` — DAO / SEP auto-detection + photometry (export: `format=json|csv|fits`, `bundle=zip`)  
  - `/preview_apertures` — preview overlays with diagnostics (overlay/panel layouts, PNG or ZIP bundle)
- **Smoke test suite (8 tests)** — ensures API stability and response consistency  
- **Prometheus-compatible metrics** — `ac_http_requests_total`, latency histograms, etc.


### 1️⃣ Install dependencies
```bash
git clone https://github.com/zhenya-klpv/astroclassify.git
cd astroclassify
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2️⃣ Run API locally

``` bash
uvicorn astroclassify.api.main:app --host 0.0.0.0 --port 8000
```
Then visit http://127.0.0.1:8000/docs for the interactive Swagger UI.

🔭 Example Usage
Auto-detection + photometry
``` bash
curl -F "file=@tiny.fits" \
  "http://127.0.0.1:8000/detect_auto?detector=sep&threshold_sigma=2.5&max_sources=50"
```

Manual photometry by coordinates
``` bash
curl -F "file=@image.png" \
  "http://127.0.0.1:8000/detect_sources?xy=120,80&xy=200,150&r=5&r_in=8&r_out=12"
```

Generate aperture preview with diagnostics
``` bash
curl -o preview.png -F "file=@image.png" \
  "http://127.0.0.1:8000/preview_apertures?xy=120,80&r=5&r_in=8&r_out=12&layout=panel&plots=radial,growth"
```

Export photometry table (CSV)
``` bash
curl -o photometry.csv -F "file=@image.png" \
  "http://127.0.0.1:8000/detect_sources?xy=120,80&r=5&format=csv&download=true"
```

📊 Metrics

Prometheus-compatible metrics are available at:
``` bash
GET /metrics
```
Example excerpt:
```
# HELP ac_http_requests_total Total HTTP requests processed
# TYPE ac_http_requests_total counter
ac_http_requests_total{method="POST",endpoint="/detect_auto"} 12
```
🧪 Tests

To verify API stability:
``` bash
python -m pytest -q tests/test_api_smoke.py
```
This runs 8 smoke tests covering:

/health

/metrics

/classify & /classify_batch

/detect_sources

/detect_auto (DAO / SEP)

/preview_apertures

- `layout=overlay|panel` — overlay keeps the base frame and optional mini-plots; panel builds a composite figure with preview + up to four diagnostics.
- `plots=` — choose diagnostics (`radial`, `growth`, `background`, `snr`, `all`, `none`).
- `bundle=zip` — download `preview.png`, `plots.png`, and `metrics.json` together (PNG remains default).
- `profile_max_r`, `percentile_low`, `percentile_high`, `stretch` — control profile depth and display stretch (linear/log/asinh).
- `labels=true|false` — toggle overlay captions (`plots`, `layout`, `count_positions`) and per-position IDs.

upload size limits

🧱 Architecture Overview
``` bash
astroclassify/
├── api/
│   ├── main.py            ← FastAPI app, routes, Prometheus registry
│   └── photometry.py      ← aperture & simple brightness modes
├── core/
│   └── device.py          ← device pick + semaphore
├── tests/
│   └── test_api_smoke.py  ← smoke test suite
└── data/                  ← sample FITS / PNG files
```

🧩 Roadmap
Milestone	Description
✅ v0.9.0	Prometheus + Photometry core
⏳ v1.0.0	CLI, Docker, GitHub Actions CI, documentation polish
💡 v1.1.0	Web dashboard + async endpoints

🧰 For Developers
Style & Linting
``` bash
pip install black isort flake8 pre-commit
pre-commit install
```
Formatting and lint checks are run automatically on commit.
Run locally with reload
``` bash
uvicorn astroclassify.api.main:app --reload

```
📄 License

This project is released under the MIT License.

Author: @zhenya-klpv
💫 AstroClassify — Open, modular, and observatory-grade photometry engine.




