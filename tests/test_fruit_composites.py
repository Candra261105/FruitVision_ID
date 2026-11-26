from __future__ import annotations

import json
import random
from pathlib import Path
from PIL import Image
import pytest

from research.procedural_backgrounds import generate_texture, STYLES
from research.generate_fruit_composites import (
    apply_feathering,
    generate_composites,
    CLASS_WEIGHTS,
)


def test_generate_texture_creates_valid_procedural_backgrounds():
    rng = random.Random(42)
    for style in STYLES:
        bg = generate_texture(width=200, height=200, style=style, rng=rng)
        assert isinstance(bg, Image.Image)
        assert bg.size == (200, 200)
        assert bg.mode == "RGB"


def test_apply_feathering_adds_alpha_transparency():
    # Solid red crop
    crop = Image.new("RGB", (50, 50), color=(255, 0, 0))
    feathered = apply_feathering(crop, radius=1.5)

    assert feathered.mode == "RGBA"
    assert feathered.size == (50, 50)

    # Center should be fully opaque (alpha 255)
    center_alpha = feathered.getpixel((25, 25))[3]
    assert center_alpha == 255

    # Corner pixel should be transparent or semi-transparent (alpha < 255)
    corner_alpha = feathered.getpixel((0, 0))[3]
    assert corner_alpha < 255


def test_rotation_in_rgba_has_transparent_corners():
    # Test that rotating an RGBA image leaves expanded corner pixels with alpha 0 (not black opaque)
    crop = Image.new("RGBA", (40, 40), (255, 0, 0, 255))
    rotated = crop.rotate(45, expand=True, resample=Image.Resampling.BILINEAR)
    assert rotated.getpixel((0, 0))[3] == 0


def test_generate_composites_creates_valid_yolo_labels(tmp_path: Path):
    # Setup mock crops and manifest
    crops_dir = tmp_path / "crops"
    manifest_path = tmp_path / "harvest_manifest.json"
    crops_dir.mkdir(parents=True)

    crops_list = []
    # Create 3 lemon, 3 tomato, 3 pumpkin crops
    class_map = {0: "tomato", 1: "pumpkin", 2: "lemon"}
    for cid, cname in class_map.items():
        cdir = crops_dir / cname
        cdir.mkdir()
        for idx in range(3):
            cimg = Image.new("RGB", (60, 60), color=(idx * 40, cid * 80, 100))
            cpath = cdir / f"{cname}_{idx}.jpg"
            cimg.save(cpath)
            crops_list.append(
                {
                    "crop_path": str(cpath),
                    "class_id": cid,
                    "class_name": cname,
                    "width": 60,
                    "height": 60,
                    "source_image": "mock.jpg",
                }
            )

    manifest_data = {
        "total_crops": len(crops_list),
        "crops": crops_list,
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    out_dir = tmp_path / "expanded"
    result = generate_composites(
        manifest_path=manifest_path,
        output_dir=out_dir,
        count=5,
        canvas_size=(640, 640),
        seed=1234,
    )

    assert result["count"] == 5
    img_dir = out_dir / "images" / "train"
    lbl_dir = out_dir / "labels" / "train"
    assert img_dir.exists()
    assert lbl_dir.exists()

    generated_images = list(img_dir.glob("*.jpg"))
    generated_labels = list(lbl_dir.glob("*.txt"))
    assert len(generated_images) == 5
    assert len(generated_labels) == 5

    # Verify each label file has valid bounding boxes
    for lbl in generated_labels:
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            cid = int(parts[0])
            assert cid in (0, 1, 2)
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < bw <= 1.0
            assert 0.0 < bh <= 1.0
            # Ensure box fits inside canvas
            assert cx - bw / 2 >= -0.05
            assert cx + bw / 2 <= 1.05
            assert cy - bh / 2 >= -0.05
            assert cy + bh / 2 <= 1.05
