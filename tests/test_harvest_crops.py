from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

from research.harvest_fruit_crops import (
    harvest_from_yolo,
    ingest_existing_crops,
    CLASS_NAMES,
)


def test_harvest_from_yolo_extracts_crops_with_padding(tmp_path: Path):
    # Setup mock image 100x100
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    out_dir = tmp_path / "crops"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_dir / "sample.jpg")

    # YOLO box: class 14 (lemon in bangladesh map), center 0.5, 0.5, width 0.4, height 0.4 -> 30 to 70
    (lbl_dir / "sample.txt").write_text("14 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    class_map = {14: 2}  # 14 -> lemon (class_id 2)
    records = harvest_from_yolo(img_dir, lbl_dir, class_map, out_dir, padding=4, min_size=10)

    assert len(records) == 1
    rec = records[0]
    assert rec["class_id"] == 2
    assert rec["class_name"] == "lemon"
    crop_path = Path(rec["crop_path"])
    assert crop_path.exists()

    with Image.open(crop_path) as c:
        # width = 40 + 2*4 = 48, height = 40 + 2*4 = 48
        assert c.size == (48, 48)


def test_harvest_ignores_tiny_crops(tmp_path: Path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    out_dir = tmp_path / "crops"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img.save(img_dir / "tiny.jpg")

    # Box size 0.05 x 0.05 -> 5x5 px (less than min_size 12)
    (lbl_dir / "tiny.txt").write_text("20 0.5 0.5 0.05 0.05\n", encoding="utf-8")

    class_map = {20: 0}  # 20 -> tomato
    records = harvest_from_yolo(img_dir, lbl_dir, class_map, out_dir, padding=2, min_size=12)

    assert len(records) == 0


def test_ingest_existing_crops(tmp_path: Path):
    src_dir = tmp_path / "existing_crops"
    out_dir = tmp_path / "crops"
    src_dir.mkdir(parents=True)

    img = Image.new("RGB", (64, 64), color=(200, 50, 50))
    img.save(src_dir / "crop1.jpg")

    records = ingest_existing_crops(src_dir, class_id=0, class_name="tomato", output_dir=out_dir, min_size=16)

    assert len(records) == 1
    assert records[0]["class_id"] == 0
    assert records[0]["class_name"] == "tomato"
    assert Path(records[0]["crop_path"]).exists()


def test_harvest_respects_exclude_stems(tmp_path: Path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    out_dir = tmp_path / "crops"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Image 1 is in test/val split (excluded)
    img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img1.save(img_dir / "val_img_01.jpg")
    (lbl_dir / "val_img_01.txt").write_text("14 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    # Image 2 is in train split (allowed)
    img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img2.save(img_dir / "train_img_01.jpg")
    (lbl_dir / "train_img_01.txt").write_text("14 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    class_map = {14: 2}
    exclude = {"val_img_01"}
    records = harvest_from_yolo(img_dir, lbl_dir, class_map, out_dir, padding=2, min_size=10, exclude_stems=exclude)

    assert len(records) == 1
    assert "train_img_01" in records[0]["source_image"]
    assert "val_img_01" not in records[0]["source_image"]

