from __future__ import annotations

import io
import zipfile
from pathlib import Path
from PIL import Image
import pytest

from research.ai_refine_crops import (
    refine_and_feather_crop,
    extract_tomato_ripeness_crops,
)


def test_refine_and_feather_crop():
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    refined = refine_and_feather_crop(img, feather_radius=3)
    assert refined.mode == "RGBA"
    assert refined.size == (100, 100)
    alpha = refined.split()[-1]
    # Check that corner pixel is fully transparent (anti-corner artifact)
    assert alpha.getpixel((0, 0)) == 0
    # Check that center pixel is fully opaque
    assert alpha.getpixel((50, 50)) == 255


def test_extract_tomato_ripeness_crops(tmp_path: Path):
    zip_path = tmp_path / "dummy_tomatoes.zip"
    out_dir = tmp_path / "crops"

    # Create dummy zip with simulated tomato crops
    with zipfile.ZipFile(zip_path, "w") as z:
        for idx in range(10):
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (200, 20, 20)).save(buf, format="JPEG")
            z.writestr(f"cls_tomato/train/ripe/tom_{idx}.jpg", buf.getvalue())

    records = extract_tomato_ripeness_crops(zip_path, out_dir, limit=5)
    assert len(records) == 5
    assert (out_dir / "tomato").is_dir()
    saved_files = list((out_dir / "tomato").glob("*.png")) + list((out_dir / "tomato").glob("*.jpg"))
    assert len(saved_files) == 5
