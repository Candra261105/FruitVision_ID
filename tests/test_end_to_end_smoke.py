from __future__ import annotations

from pathlib import Path
import zipfile
import yaml
import pytest


def test_zip_archive_integrity():
    zip_path = Path("data/tomato-pumpkin-lemon-expanded.zip")
    if not zip_path.exists():
        pytest.skip("tomato-pumpkin-lemon-expanded.zip has not been built in this environment yet")
    assert zip_path.stat().st_size > 50 * 1024 * 1024, "Zip file should be over 50MB"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()

        # Check critical files
        assert "dataset/data.yaml" in namelist
        assert "training/colab_train_expanded.py" in namelist

        # Check split counts in zip
        val_images = [n for n in namelist if n.startswith("dataset/images/val/") and n.endswith(".jpg")]
        test_images = [n for n in namelist if n.startswith("dataset/images/test/") and n.endswith(".jpg")]
        train_images = [n for n in namelist if n.startswith("dataset/images/train/") and n.endswith(".jpg")]

        assert len(val_images) == 300, f"Expected exactly 300 val images, got {len(val_images)}"
        assert len(test_images) == 300, f"Expected exactly 300 test images, got {len(test_images)}"
        assert len(train_images) >= 3000, f"Expected >= 3000 train images, got {len(train_images)}"

        # Validate YAML content inside zip
        yaml_data = yaml.safe_load(zf.read("dataset/data.yaml").decode("utf-8"))
        assert yaml_data["names"] == ["tomato", "pumpkin", "lemon"]


def test_colab_guide_documentation_exists():
    guide_path = Path("docs/COLAB_EXPANDED_GUIDE.md")
    assert guide_path.exists(), "Colab guide must exist"
    content = guide_path.read_text(encoding="utf-8")
    assert "GPU T4" in content
    assert "tomato-pumpkin-lemon-expanded.zip" in content
    assert "best.pt" in content
    assert "imgsz=800" in content or "800" in content
