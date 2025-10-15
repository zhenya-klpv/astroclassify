from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency
    from astropy.table import Table  # type: ignore

    _HAS_TABLE = True
except Exception:  # pragma: no cover - soft dependency
    Table = None  # type: ignore
    _HAS_TABLE = False


@dataclass(frozen=True)
class ColumnSpec:
    header: str
    aliases: Tuple[str, ...]
    kind: str = "float"  # float|int|mixed


BASE_COLUMNS: Tuple[ColumnSpec, ...] = (
    ColumnSpec("x", ("x",)),
    ColumnSpec("y", ("y",)),
    ColumnSpec("r", ("r", "radius")),
    ColumnSpec("aperture_sum", ("aperture_sum",)),
    ColumnSpec("bkg_mean", ("bkg_mean", "background_mean")),
    ColumnSpec("bkg_area", ("bkg_area", "background_area")),
    ColumnSpec("flux_sub", ("flux_sub", "flux_subtracted")),
)

OPTIONAL_COLUMNS: Tuple[ColumnSpec, ...] = (
    ColumnSpec("flux_err", ("flux_err", "flux_error")),
    ColumnSpec("bkg_rms", ("bkg_rms", "background_rms")),
    ColumnSpec("SNR", ("SNR", "snr")),
    ColumnSpec("flags", ("flags",), kind="int"),
    ColumnSpec("fwhm", ("fwhm", "fwhm_pix")),
    ColumnSpec("ellipticity", ("ellipticity", "ellip")),
    ColumnSpec("position_angle", ("position_angle", "pa")),
    ColumnSpec("apcorr", ("apcorr", "aperture_correction")),
)


@dataclass
class ExportArtifact:
    content: bytes
    media_type: str
    extension: str
    columns: List[str]


def _normalize_value(value: Any, kind: str) -> Any:
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if kind == "int":
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return value

    if isinstance(value, (int, np.integer)):
        return float(value)

    if isinstance(value, (float, np.floating)):
        return float(value)

    return value


def _value_from_aliases(row: Mapping[str, Any], aliases: Tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in row:
            return row[alias]
        alt = alias.lower()
        if alt in row:
            return row[alt]
        alt_upper = alias.upper()
        if alt_upper in row:
            return row[alt_upper]
    return None


def _prepare_rows(results: Sequence[Mapping[str, Any]]) -> Tuple[List[ColumnSpec], List[Dict[str, Any]]]:
    columns: List[ColumnSpec] = list(BASE_COLUMNS)
    for spec in OPTIONAL_COLUMNS:
        if any(_value_from_aliases(row, spec.aliases) is not None for row in results):
            columns.append(spec)

    prepared: List[Dict[str, Any]] = []
    for row in results:
        item: Dict[str, Any] = {}
        for spec in columns:
            value = _value_from_aliases(row, spec.aliases)
            item[spec.header] = _normalize_value(value, spec.kind)
        prepared.append(item)
    return columns, prepared


def export_photometry(
    results: Sequence[Mapping[str, Any]],
    fmt: str,
    *,
    csv_delimiter: str = ",",
    csv_float_fmt: str = ".6f",
    json_indent: Optional[int] = 2,
    json_compact: bool = False,
) -> ExportArtifact:
    fmt = fmt.lower()
    columns, rows = _prepare_rows(results)
    headers = [spec.header for spec in columns]

    if fmt == "csv":
        content = _to_csv(rows, headers, delimiter=csv_delimiter, float_fmt=csv_float_fmt)
        return ExportArtifact(content=content, media_type="text/csv", extension="csv", columns=headers)

    if fmt == "fits":
        if not _HAS_TABLE or Table is None:
            raise RuntimeError("FITS export requires astropy.table to be installed.")
        content = _to_fits(rows, headers)
        return ExportArtifact(content=content, media_type="application/fits", extension="fits", columns=headers)

    if fmt == "json":
        content = _to_json(rows, indent=None if json_compact else json_indent, compact=json_compact)
        return ExportArtifact(content=content, media_type="application/json", extension="json", columns=headers)

    raise ValueError(f"Unsupported export format: {fmt}")


def _format_float(value: Any, float_fmt: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return format(float(value), float_fmt)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return format(float(value), float_fmt)
    return str(value)


def _to_csv(
    rows: Sequence[Mapping[str, Any]],
    headers: Sequence[str],
    *,
    delimiter: str,
    float_fmt: str,
) -> bytes:
    if not delimiter:
        delimiter = ","
    if delimiter == "\\t" or delimiter.lower() == "tab":
        delimiter = "\t"
    if len(delimiter) != 1:
        delimiter = delimiter[0]

    sio = io.StringIO()
    writer = csv.writer(sio, delimiter=delimiter)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([_format_float(row.get(key), float_fmt) for key in headers])
    return sio.getvalue().encode("utf-8")


def _to_json(
    rows: Sequence[Mapping[str, Any]],
    *,
    indent: Optional[int],
    compact: bool,
) -> bytes:
    separators = (",", ":") if compact else None
    payload = json.dumps(rows, ensure_ascii=False, indent=indent, separators=separators)
    return payload.encode("utf-8")


def _to_fits(rows: Sequence[Mapping[str, Any]], headers: Sequence[str]) -> bytes:
    table_rows = [{key: row.get(key) for key in headers} for row in rows]
    buf = io.BytesIO()
    table = Table(rows=table_rows, names=list(headers))
    table.write(buf, format="fits", overwrite=True)
    return buf.getvalue()


def build_zip_bundle(
    artifact: ExportArtifact,
    *,
    metadata: Mapping[str, Any],
    filename_stem: str,
    preview_png: Optional[bytes] = None,
    json_payload: Optional[bytes] = None,
) -> bytes:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{filename_stem}.{artifact.extension}", artifact.content)
        meta = dict(metadata)
        meta.setdefault("columns", artifact.columns)
        meta.setdefault("count", len(metadata.get("results", [])) if "results" in metadata else None)
        zf.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        if preview_png is not None:
            zf.writestr("preview.png", preview_png)
        if json_payload is not None:
            zf.writestr("data.json", json_payload)
    zip_buf.seek(0)
    return zip_buf.getvalue()
