from __future__ import annotations

from typing import List

import pytest

try:  # optional dependency guard
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - fastapi not installed
    TestClient = None  # type: ignore

try:
    from astroclassify.api.main import app
except Exception:  # pragma: no cover - fastapi not installed
    app = None  # type: ignore
from astroclassify.data_sources import CutoutResult, CutoutError


if TestClient is not None and app is not None:
    client = TestClient(app)
else:  # pragma: no cover - fastapi missing
    client = None  # type: ignore


class _StubProvider:
    name = "stub"

    def __init__(self, result: CutoutResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    def fetch(self, request):  # pragma: no cover - simple stub
        if self._error:
            raise self._error
        return self._result or CutoutResult(b"data", "image/fits", "stub.fits")


def test_cutout_success(monkeypatch):
    if client is None:
        pytest.skip("fastapi test client not available")
    monkeypatch.setattr(
        "astroclassify.api.main.get_cutout_provider",
        lambda service: _StubProvider(CutoutResult(b"fitsdata", "image/fits", "stub.fits")),
    )
    resp = client.get("/v1/cutout?service=sdss&ra=13.4&dec=-2.1&size_arcsec=60")
    assert resp.status_code == 200
    assert resp.content == b"fitsdata"
    assert resp.headers["content-type"] == "image/fits"
    assert resp.headers["x-astro-provider"] == "stub"


def test_cutout_provider_error(monkeypatch):
    if client is None:
        pytest.skip("fastapi test client not available")
    monkeypatch.setattr(
        "astroclassify.api.main.get_cutout_provider",
        lambda service: _StubProvider(error=CutoutError("no data")),
    )
    resp = client.get("/v1/cutout?service=sdss&ra=13.4&dec=-2.1&size_arcsec=60")
    assert resp.status_code == 404


def test_cutout_invalid_service():
    if client is None:
        pytest.skip("fastapi test client not available")
    resp = client.get("/v1/cutout?service=unknown&ra=10&dec=20&size_arcsec=30")
    assert resp.status_code == 400
