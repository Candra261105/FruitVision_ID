from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest
import yaml

from research.build_expanded_dataset import (
    freeze_benchmark_splits,
    integrate_train_data,
    verify_dataset_leakage,
    write_dataset_yaml,
    CLASS_NAMES,
)


def test_freeze_benchmark_splits_copies_exact_val_test(tmp_path: Path):
    # Setup mock source dataset
    src_dir = tmp_path / "benchmark"
    dst_dir = tmp_path / "expanded"

    for split in ("val", "test"):
        (src_dir / "images" / split).mkdir(parents=True)
        (src_dir / "labels" / split).mkdir(parents=True)

        for i in range(5):
            img = Image.new("RGB", (50, 50), color=(i * 10, i * 20, 30))
            img.save(src_dir / "images" / split / f"bench_{i:03d}.jpg")
            (src_dir / "labels" / split / f"bench_{i:03d}.txt").write_text(f"0 0.5 0.5 0.2 0.2\n")

    report = freeze_benchmark_splits(src_dir, dst_dir)
    assert report["val_count"] == 5
    assert report["test_count"] == 5

    # Verify destination files
    for split in ("val", "test"):
        assert len(list((dst_dir / "images" / split).glob("*.jpg"))) == 5
        assert len(list((dst_dir / "labels" / split).glob("*.txt"))) == 5


def test_verify_dataset_leakage_catches_synthetic_in_val(tmp_path: Path):
    dst_dir = tmp_path / "expanded"
    (dst_dir / "images" / "val").mkdir(parents=True)
    (dst_dir / "labels" / "val").mkdir(parents=True)
    (dst_dir / "images" / "train").mkdir(parents=True)
    (dst_dir / "labels" / "train").mkdir(parents=True)

    # Valid train synthetic
    img1 = Image.new("RGB", (50, 50))
    img1.save(dst_dir / "images" / "train" / "synthetic_01.jpg")
    (dst_dir / "labels" / "train" / "synthetic_01.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Real val image
    img2 = Image.new("RGB", (50, 50))
    img2.save(dst_dir / "images" / "val" / "real_01.jpg")
    (dst_dir / "labels" / "val" / "real_01.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    # Clean dataset should have no leakage
    assert verify_dataset_leakage(dst_dir) is True

    # Inject forbidden synthetic image into val
    img3 = Image.new("RGB", (50, 50))
    img3.save(dst_dir / "images" / "val" / "synthetic_leaked.jpg")
    (dst_dir / "labels" / "val" / "synthetic_leaked.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    # Should detect leakage
    assert verify_dataset_leakage(dst_dir) is False


def test_write_dataset_yaml_structure(tmp_path: Path):
    target = tmp_path / "dataset"
    target.mkdir()
    yaml_path = write_dataset_yaml(target)

    assert yaml_path.exists()
    content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert content["path"] == "dataset"
    assert content["train"] == "images/train"
    assert content["val"] == "images/val"
    assert content["test"] == "images/test"
    assert content["names"] == ["tomato", "pumpkin", "lemon"]


def test_integrate_train_data_with_extra_real_sources(tmp_path: Path):
    src_dir = tmp_path / "base"
    comp_dir = tmp_path / "composites"
    extra_dir = tmp_path / "extra_real"
    target_dir = tmp_path / "output"

    # Base train
    (src_dir / "images" / "train").mkdir(parents=True)
    (src_dir / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(src_dir / "images" / "train" / "base_01.jpg")
    (src_dir / "labels" / "train" / "base_01.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Composites
    (comp_dir / "images" / "train").mkdir(parents=True)
    (comp_dir / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(comp_dir / "images" / "train" / "synthetic_01.jpg")
    (comp_dir / "labels" / "train" / "synthetic_01.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    # Extra real
    (extra_dir / "images" / "train").mkdir(parents=True)
    (extra_dir / "labels" / "train").mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(extra_dir / "images" / "train" / "kitchen_01.jpg")
    (extra_dir / "labels" / "train" / "kitchen_01.txt").write_text("2 0.5 0.5 0.2 0.2\n")

    stats = integrate_train_data(
        source_dataset=src_dir,
        composite_dir=comp_dir,
        target_dataset=target_dir,
        extra_real_dirs=[extra_dir],
    )
    assert stats["real_train"] == 2
    assert stats["synthetic_train"] == 1
    assert (target_dir / "images" / "train" / "kitchen_01.jpg").is_file()
    assert (target_dir / "labels" / "train" / "kitchen_01.txt").is_file()
