from pathlib import Path
import pytest

from research.download_production_model import (
    MODEL_FILENAME,
    MODEL_URL,
    ensure_production_model,
)


def test_model_downloader_metadata():
    assert "Piyu12/fruit-veg-yolo11m-detector" in MODEL_URL
    assert MODEL_FILENAME == "fruit_veg_yolo11m.pt"


def test_ensure_production_model_uses_existing_file(tmp_path):
    target = tmp_path / "models" / "fruit_veg_yolo11m.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"dummy_weights" * 100)

    # Should immediately return existing file without download
    result = ensure_production_model(target)
    assert result == target
    assert result.is_file()
