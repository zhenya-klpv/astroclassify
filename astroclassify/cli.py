from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from urllib import error as urllib_error
from urllib import parse, request


DEFAULT_BASE_URL = os.environ.get("ASTRO_API_URL", "http://127.0.0.1:8000/v1")


@dataclass
class PreparedRequest:
    method: str
    url: str
    params: List[tuple[str, str]]
    file_path: Path
    description: str


@dataclass
class SimpleResponse:
    status: int
    headers: Dict[str, str]
    content: bytes

    def json(self) -> Dict[str, object]:
        return json.loads(self.content.decode("utf-8"))


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def _format_xy(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        parts = value.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid xy value: {value!r}")
        try:
            float(parts[0])
            float(parts[1])
        except ValueError as exc:  # pragma: no cover - argparse will report message
            raise argparse.ArgumentTypeError(f"Invalid xy value: {value!r}") from exc
        out.append(f"{parts[0]},{parts[1]}")
    if not out:
        raise argparse.ArgumentTypeError("At least one --xy coordinate is required")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astrocli",
        description="CLI helper for the AstroClassify API",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ASTRO_API_TIMEOUT", "30")),
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request that would be sent and exit",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Run /v1/detect_auto and optionally download results")
    detect.add_argument("--file", required=True, type=Path, help="Path to the image (FITS/PNG/JPEG)")
    detect.add_argument("--detector", default="sep", choices=["sep", "dao"], help="Detector backend")
    detect.add_argument("--threshold", type=float, default=3.0, help="Detection sigma threshold")
    detect.add_argument("--max-sources", type=int, default=50, help="Maximum number of sources to keep")
    detect.add_argument("--format", default="json", choices=["json", "csv", "fits"], help="Export format")
    detect.add_argument("--bundle", choices=["none", "zip"], default="none", help="Bundle response into zip")
    detect.add_argument("--download", action="store_true", help="Force Content-Disposition attachment")
    detect.add_argument("--output", type=Path, help="Destination file (for binary responses)")
    detect.add_argument("--r", type=float, default=5.0, help="Aperture radius (px)")
    detect.add_argument("--r-in", type=float, default=8.0, help="Annulus inner radius (px)")
    detect.add_argument("--r-out", type=float, default=12.0, help="Annulus outer radius (px)")
    detect.add_argument("--phot-mode", default="aperture", choices=["aperture", "psf"], help="Photometry mode")
    detect.add_argument("--psf-stamp", type=int, default=8, help="PSF stamp radius (px)")
    detect.add_argument("--psf-fit", type=int, default=4, help="PSF fit radius (px)")
    detect.add_argument("--zeropoint", type=float, help="Photometric zero point")
    detect.add_argument("--exptime", type=float, help="Exposure time in seconds")
    detect.add_argument("--gain", type=float, help="Detector gain (e-/ADU)")
    detect.add_argument("--mag-system", default=None, help="Magnitude system label (e.g. AB or Vega)")

    preview = sub.add_parser("preview", help="Render /v1/preview_apertures overlay or bundle")
    preview.add_argument("--file", required=True, type=Path, help="Path to the image (FITS/PNG/JPEG)")
    preview.add_argument(
        "--xy",
        action="append",
        required=True,
        help="Coordinate pair 'x,y'. Pass multiple times for multiple apertures.",
    )
    preview.add_argument("--r", type=float, default=5.0, help="Aperture radius (px)")
    preview.add_argument("--r-in", type=float, default=8.0, help="Annulus inner radius (px)")
    preview.add_argument("--r-out", type=float, default=12.0, help="Annulus outer radius (px)")
    preview.add_argument(
        "--layout",
        default="overlay",
        choices=["overlay", "panel", "grid", "row"],
        help="Preview layout",
    )
    preview.add_argument(
        "--bundle",
        default="png",
        choices=["png", "zip"],
        help="Output format",
    )
    preview.add_argument("--plots", default="radial,growth", help="Comma-separated diagnostic plots")
    preview.add_argument("--output", type=Path, help="Destination file for rendered preview/bundle")

    return parser


def _prepare_detect(args: argparse.Namespace) -> PreparedRequest:
    params = [
        ("detector", args.detector),
        ("threshold_sigma", f"{args.threshold}"),
        ("max_sources", str(args.max_sources)),
        ("format", args.format),
        ("bundle", args.bundle),
        ("download", "true" if args.download else "false"),
        ("r", f"{args.r}"),
        ("r_in", f"{args.r_in}"),
        ("r_out", f"{args.r_out}"),
        ("phot_mode", args.phot_mode),
        ("psf_stamp", str(args.psf_stamp)),
        ("psf_fit", str(args.psf_fit)),
    ]
    if args.zeropoint is not None:
        params.append(("zeropoint", f"{args.zeropoint}"))
    if args.exptime is not None:
        params.append(("exptime", f"{args.exptime}"))
    if args.gain is not None:
        params.append(("gain", f"{args.gain}"))
    if args.mag_system:
        params.append(("mag_system", args.mag_system))
    url = f"{args.base_url.rstrip('/')}/detect_auto"
    desc = "Detect auto sources"
    return PreparedRequest("POST", url, params, args.file, desc)


def _prepare_preview(args: argparse.Namespace) -> PreparedRequest:
    xy_values = _format_xy(args.xy)
    params: List[tuple[str, str]] = [
        ("r", f"{args.r}"),
        ("r_in", f"{args.r_in}"),
        ("r_out", f"{args.r_out}"),
        ("layout", args.layout),
        ("plots", args.plots),
        ("bundle", args.bundle),
    ]
    for value in xy_values:
        params.append(("xy", value))
    url = f"{args.base_url.rstrip('/')}/preview_apertures"
    desc = "Render aperture preview"
    return PreparedRequest("POST", url, params, args.file, desc)


def _print_dry_run(prep: PreparedRequest) -> None:
    grouped: Dict[str, List[str]] = {}
    for key, value in prep.params:
        grouped.setdefault(key, []).append(value)
    params_repr = json.dumps(grouped, indent=2)
    print(f"{prep.method} {prep.url}")
    print(f"File: {prep.file_path}")
    print(f"Params: {params_repr}")


def _encode_multipart(file_path: Path) -> Tuple[bytes, str]:
    boundary = f"----astrocli{uuid.uuid4().hex}"
    mime = _guess_mime(file_path)
    file_bytes = file_path.read_bytes()
    lines: List[bytes] = [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode(),
        f"Content-Type: {mime}".encode(),
        b"",
        file_bytes,
        f"--{boundary}--".encode(),
        b"",
    ]
    body = b"\r\n".join(lines)
    return body, boundary


def _send_request(prep: PreparedRequest, timeout: float) -> SimpleResponse:
    file_path = prep.file_path
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    query = parse.urlencode(prep.params, doseq=True)
    url = prep.url
    if query:
        url = f"{url}?{query}"

    body, boundary = _encode_multipart(file_path)
    req = request.Request(url, data=body, method=prep.method)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            headers = {k.lower(): v for k, v in resp.getheaders()}
            return SimpleResponse(resp.status, headers, content)
    except urllib_error.HTTPError as exc:
        error_body = exc.read()
        message = error_body.decode("utf-8", errors="ignore") or exc.reason
        raise RuntimeError(f"HTTP {exc.code}: {message}")


def _derive_output_path(args: argparse.Namespace, response: SimpleResponse, default_name: str) -> Path:
    if args.output:
        return args.output
    disposition = response.headers.get("content-disposition", "")
    if "filename=" in disposition:
        filename = disposition.split("filename=")[-1].strip('"')
        return Path(filename)
    return Path(default_name)


def _handle_response(args: argparse.Namespace, prep: PreparedRequest, response: SimpleResponse) -> int:
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    # Binary responses
    default_name = "preview.png" if "preview" in prep.desc.lower() else "detect.auto.bin"
    output_path = _derive_output_path(args, response, default_name)
    output_path.write_bytes(response.content)
    print(f"Saved response to {output_path.resolve()}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "detect":
        prep = _prepare_detect(args)
    elif args.command == "preview":
        prep = _prepare_preview(args)
    else:  # pragma: no cover - argparse enforces options
        parser.error("Unsupported command")
        return 1

    if args.dry_run:
        _print_dry_run(prep)
        return 0

    try:
        response = _send_request(prep, timeout=args.timeout)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"astrocli: {exc}", file=sys.stderr)
        return 2

    return _handle_response(args, prep, response)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
