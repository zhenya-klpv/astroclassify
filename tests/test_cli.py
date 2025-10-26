from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from astroclassify import cli


def _write_test_image(tmp_path: Path) -> Path:
    from PIL import Image

    data = np.linspace(0, 255, num=64, dtype=np.uint8).reshape(8, 8)
    image = Image.fromarray(data, mode="L")
    path = tmp_path / "frame.png"
    with path.open("wb") as fh:
        image.save(fh, format="PNG")
    return path


def test_detect_dry_run(tmp_path: Path, capsys) -> None:
    image_path = _write_test_image(tmp_path)
    rc = cli.main(
        [
            "--dry-run",
            "--base-url",
            "http://example.test/v1",
            "detect",
            "--file",
            str(image_path),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "POST http://example.test/v1/detect_auto" in captured.out
    assert str(image_path) in captured.out


def test_preview_dry_run_requires_xy(tmp_path: Path) -> None:
    image_path = _write_test_image(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["preview", "--file", str(image_path)])


def test_preview_dry_run_output(tmp_path: Path, capsys) -> None:
    image_path = _write_test_image(tmp_path)
    rc = cli.main(
        [
            "--dry-run",
            "preview",
            "--file",
            str(image_path),
            "--xy",
            "10,11",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "preview_apertures" in captured.out
    assert "10,11" in captured.out
