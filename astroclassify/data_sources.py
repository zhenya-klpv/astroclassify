from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import requests

try:  # optional dependency for SIA parsing
    from astropy.io import votable  # type: ignore

    _HAS_VOTABLE = True
except Exception:  # pragma: no cover - astropy not installed
    votable = None  # type: ignore
    _HAS_VOTABLE = False

try:  # optional dependency
    from astroquery.mast import Cutouts  # type: ignore
    from astroquery.mast import Observations  # type: ignore
    from astropy import units as u  # type: ignore

    _HAS_ASTROQUERY = True
except Exception:  # pragma: no cover - astroquery not installed
    Cutouts = None  # type: ignore
    Observations = None  # type: ignore
    u = None  # type: ignore
    _HAS_ASTROQUERY = False

__all__ = [
    "CutoutRequest",
    "CutoutResult",
    "CutoutProvider",
    "get_cutout_provider",
]


class CutoutError(RuntimeError):
    """Raised when a cutout request fails."""


@dataclass
class CutoutRequest:
    ra: float
    dec: float
    size_deg: Optional[float] = None
    size_arcmin: Optional[float] = None
    size_arcsec: Optional[float] = None
    band: Optional[str] = None
    filters: Optional[Sequence[str]] = None


@dataclass
class CutoutResult:
    content: bytes
    media_type: str
    filename: str


class CutoutProvider(Protocol):
    name: str

    def fetch(self, request: CutoutRequest) -> CutoutResult:
        ...


def _http_get(url: str, params: Optional[Dict[str, str]] = None, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


class NoirlabSIAProvider:
    name = "noirlab"
    BASE_URL = "https://datalab.noirlab.edu/sia/nsc_dr2"

    def fetch(self, request: CutoutRequest) -> CutoutResult:
        if not _HAS_VOTABLE:
            raise NotImplementedError("astropy is required for NOIRLab SIA parsing.")
        size_deg = request.size_deg or (request.size_arcmin or 1.0) / 60.0
        params = {
            "POS": f"{request.ra},{request.dec}",
            "SIZE": f"{size_deg:.6f}",
            "FORMAT": "image/fits",
        }
        if request.band:
            params["BAND"] = request.band
        table_resp = _http_get(self.BASE_URL, params=params)
        access_url = self._extract_access_url(table_resp.content)
        if not access_url:
            raise CutoutError("No cutout available for the specified region.")
        fits_resp = _http_get(access_url, timeout=90)
        return CutoutResult(
            content=fits_resp.content,
            media_type="image/fits",
            filename="noirlab_nsc_cutout.fits",
        )

    @staticmethod
    def _extract_access_url(payload: bytes) -> Optional[str]:
        if not _HAS_VOTABLE or votable is None:
            return None
        try:
            table = votable.parse_single_table(io.BytesIO(payload))
            arr = table.array
            if arr is None or len(arr) == 0:
                return None
            first = arr[0]
            if "access_url" in first.dtype.names:
                return str(first["access_url"])
        except Exception:
            return None
        return None


class SdssCutoutProvider:
    name = "sdss"
    BASE_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"

    def fetch(self, request: CutoutRequest) -> CutoutResult:
        size_arcsec = request.size_arcsec or (request.size_arcmin or 1.0) * 60.0
        scale = 0.396127  # arcsec per pixel
        pixels = int(math.ceil(size_arcsec / scale))
        params = {
            "ra": f"{request.ra}",
            "dec": f"{request.dec}",
            "scale": f"{scale}",
            "width": str(max(32, min(pixels, 2048))),
            "height": str(max(32, min(pixels, 2048))),
        }
        resp = _http_get(self.BASE_URL, params=params)
        return CutoutResult(
            content=resp.content,
            media_type="image/jpeg",
            filename="sdss_cutout.jpg",
        )


class ZtfCutoutProvider:
    name = "ztf"
    BASE_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
    DATA_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"

    def fetch(self, request: CutoutRequest) -> CutoutResult:
        size_deg = request.size_deg
        if size_deg is None:
            if request.size_arcmin is not None:
                size_deg = request.size_arcmin / 60.0
            elif request.size_arcsec is not None:
                size_deg = request.size_arcsec / 3600.0
            else:
                size_deg = 0.05
        params = {
            "POS": f"{request.ra},{request.dec}",
            "SIZE": f"{size_deg:.6f}",
            "BANDNAME": (request.band or "zg"),
            "outfmt": "json",
            "VERB": "1",
        }
        search_resp = _http_get(self.BASE_URL, params=params)
        data = search_resp.json()
        if not data or "table" not in data or not data["table"]["rows"]:
            raise CutoutError("No ZTF products found for the specified region.")
        first = data["table"]["rows"][0]
        rel_path = first.get("filefracday")
        field = first.get("field")
        ccd = first.get("ccdid")
        quad = first.get("qid")
        if None in (rel_path, field, ccd, quad):
            raise CutoutError("Incomplete ZTF metadata for cutout retrieval.")
        fits_path = f"{self.DATA_URL}/{str(field).zfill(3)}/{rel_path}/{int(ccd):02d}/{int(quad)}"
        cutout_resp = _http_get(fits_path, timeout=120)
        return CutoutResult(
            content=cutout_resp.content,
            media_type="image/fits",
            filename="ztf_cutout.fits",
        )


class MastCutoutProvider:
    name = "mast"

    def fetch(self, request: CutoutRequest) -> CutoutResult:
        if not _HAS_ASTROQUERY:
            raise NotImplementedError("astroquery is required for MAST cutouts.")
        radius = request.size_arcmin or (request.size_deg or 0.05) * 60.0
        obs = Observations.query_region(
            f"{request.ra} {request.dec}",
            radius=radius * u.arcmin,
        )
        if len(obs) == 0:
            raise CutoutError("No MAST observations found in the specified region.")
        filters = request.filters or []
        if filters:
            obs = obs[[str(x).lower() in {f.lower() for f in filters} for x in obs["filters"]]]
        if len(obs) == 0:
            raise CutoutError("Filtered MAST observations were empty.")
        size_arcsec = request.size_arcsec or (request.size_arcmin or 2.0) * 60.0
        size_str = f"{size_arcsec}\""
        with tempfile.TemporaryDirectory() as tmpdir:
            cutouts = Cutouts.download_cutouts(
                obs[:1],
                size=size_str,
                download_dir=tmpdir,
            )
            if not cutouts:
                raise CutoutError("MAST returned no cutouts for the specified region.")
            filepath = cutouts[0]["Local Path"]
            data = Path(filepath).read_bytes()
        return CutoutResult(
            content=data,
            media_type="image/fits",
            filename=os.path.basename(filepath),
        )


def get_cutout_provider(service: str) -> CutoutProvider:
    service_lower = service.lower()
    if service_lower in {"noirlab", "nsc", "decam"}:
        return NoirlabSIAProvider()
    if service_lower in {"sdss", "dr17"}:
        return SdssCutoutProvider()
    if service_lower in {"ztf"}:
        return ZtfCutoutProvider()
    if service_lower in {"mast", "panstarrs", "hst", "galex"}:
        return MastCutoutProvider()
    raise ValueError(f"Unsupported service: {service}")
