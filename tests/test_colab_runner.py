from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from research.colab_download_helper import safe_download_artifacts


def test_colab_download_fallback_logic():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        best_pt = tmp_path / "best.pt"
        best_pt.write_text("dummy model weights")
        results_png = tmp_path / "results.png"
        results_png.write_text("dummy results png")

        zip_out = tmp_path / "artifacts.zip"
        status = safe_download_artifacts(
            files_to_bundle=[best_pt, results_png],
            zip_dest=zip_out,
            drive_dest=tmp_path / "drive_mock",
        )

        assert status["zip_created"] is True
        assert zip_out.is_file()
        assert status["drive_copied"] is False or status["drive_copied"] is True


def test_colab_train_expanded_configuration():
    colab_script = Path("data/tomato-pumpkin-lemon-expanded/training/colab_train_expanded.py")
    assert colab_script.is_file()
    content = colab_script.read_text(encoding="utf-8")
    assert "yolo11m.pt" in content
    assert "imgsz=800" in content
    assert "safe_download_artifacts" in content
