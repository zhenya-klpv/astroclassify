import io, os, importlib, zipfile
import pytest
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No sources were found")

os.environ.setdefault("AC_MAX_UPLOAD_BYTES", "10240")  # 10 KB

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from PIL import Image
from fastapi.testclient import TestClient
from astroclassify.api.main import app

client = TestClient(app)

def _png_bytes(w=32, h=32, color=(12, 34, 56)):
    im = Image.new("RGB", (w, h), color)
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return bio.getvalue()

# ---------------------------------------------------------------------
def test_health():
    for path in ("/health", "/api/health", "/v1/health"):
        r = client.get(path)
        if r.status_code == 200:
            try:
                js = r.json()
                ok = js.get("status", "").lower() == "ok"
            except Exception:
                ok = r.text.strip().lower() == "ok"
            assert ok
            return
    pytest.fail("health endpoint not found")

def test_metrics():
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert ("# HELP" in body) or ("# TYPE" in body)
    assert len(body) > 50
    assert "astro_requests_total" in body
    assert "astro_inference_seconds_bucket" in body
    assert "astro_photometry_requests_total" in body
    assert "astro_sources_detected_total" in body

def test_classify_small_png():
    data = _png_bytes()
    files = [("files", ("tiny.png", data, "image/png"))]
    r = client.post("/classify_batch?topk=2&imagenet_norm=true", files=files)
    assert r.status_code in (200, 400), r.text
    if r.status_code == 200:
        body = r.json()
        assert body.get("count") == 1
        assert "results" in body and len(body["results"]) == 1

def test_detect_sources_simple():
    data = _png_bytes()
    r = client.post("/detect_sources", files={"file": ("ph.png", data, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    val = body.get("simple_brightness") or body.get("value")
    assert isinstance(val, (int, float)), body

@pytest.mark.skipif(importlib.util.find_spec("sep") is None, reason="sep not installed")
def test_detect_auto_sep_png():
    data = _png_bytes()
    r = client.post(
        "/detect_auto?detector=sep&threshold_sigma=1.5&max_sources=5",
        files={"file": ("img.png", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("detector") == "sep"
    assert isinstance(body.get("count", 0), int)

# --- FIXED DAO TEST ---
@pytest.mark.skipif(importlib.util.find_spec("photutils") is None, reason="photutils not installed")
def test_detect_auto_dao_png():
    data = _png_bytes()
    r = client.post(
        "/detect_auto?detector=dao&threshold_sigma=3.0&max_sources=5",
        files={"file": ("dao.png", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("detector") == "dao"
    # DAO может вернуть пустой список (NoDetectionsWarning)
    assert isinstance(body.get("count", 0), int)
    # если ничего не найдено, это ок
    if body.get("count", 0) > 0:
        assert any(k in body for k in ("positions", "fluxes", "sources")), body

@pytest.mark.skipif(
    importlib.util.find_spec("photutils") is None or importlib.util.find_spec("astropy") is None,
    reason="photometry dependencies not installed",
)
def test_detect_sources_export_csv():
    data = _png_bytes()
    r = client.post(
        "/detect_sources?xy=10,10&r=5&format=csv&download=true",
        files={"file": ("phot.png", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert "text/csv" in r.headers.get("content-type", "")
    lines = r.content.decode("utf-8").strip().splitlines()
    assert lines and "x" in lines[0] and "flux_sub" in lines[0]

@pytest.mark.skipif(
    importlib.util.find_spec("photutils") is None or importlib.util.find_spec("astropy") is None,
    reason="photometry dependencies not installed",
)
def test_detect_sources_export_zip():
    data = _png_bytes()
    r = client.post(
        "/detect_sources?xy=10,10&r=5&format=json&bundle=zip",
        files={"file": ("phot.png", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert "application/zip" in r.headers.get("content-type", "")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
        assert "metadata.json" in names
        assert any(name.endswith(".json") and name != "metadata.json" for name in names)

# --- FIXED PREVIEW TEST ---
def test_preview_apertures():
    data = _png_bytes()
    # Добавляем хотя бы одну координату (требует API)
    r = client.post(
        "/preview_apertures?xy=10,10&r=5&r_in=8&r_out=12&line=2",
        files={"file": ("prev.png", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    ct = r.headers.get("content-type", "")
    assert "image/png" in ct or "json" in ct
    if "image/png" in ct:
        assert len(r.content) > 100
        assert r.content[:8].startswith(b"\x89PNG")


def test_preview_apertures_validation_error_has_code():
    data = _png_bytes()
    r = client.post(
        "/preview_apertures?xy=10,10&r=5&r_in=4&r_out=6",
        files={"file": ("prev.png", data, "image/png")},
    )
    assert r.status_code == 400
    body = r.json()
    assert body.get("code", "").startswith("ASTRO_400")
    assert "hint" in body and body["hint"]

def test_upload_limit():
    limit = int(os.environ.get("AC_MAX_UPLOAD_BYTES", "10240"))
    big = b"x" * (limit + 1)
    r = client.post(
        "/classify_batch",
        files=[("files", ("big.bin", big, "application/octet-stream"))],
    )
    assert r.status_code in (413, 200), r.text
    if r.status_code == 200:
        body = r.json()
        assert "results" in body and body["results"], body
        assert "error" in body["results"][0], body
