from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from research.tomato_pumpkin_lemon_composites import generate_composites


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    train_images = root / "images" / "train"
    train_labels = root / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    for index in range(3):
        image = Image.new("RGB", (80, 60), (30 + index * 50, 80, 120))
        image.save(train_images / f"source_{index}.jpg")
        (train_labels / f"source_{index}.txt").write_text(
            f"{index} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    manifest = [
        {
            "output_image": f"images/train/source_{index}.jpg",
            "output_label": f"labels/train/source_{index}.txt",
            "split": "train",
            "source_id": "source",
            "source_image": str(train_images / f"source_{index}.jpg"),
            "sha256": f"hash-{index}",
            "phash": f"{index:016x}",
            "synthetic": False,
        }
        for index in range(3)
    ]
    (root / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_composites_are_deterministic_and_train_only(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    report = generate_composites(root, count=2, seed=11, max_synthetic_ratio=1.0)

    assert report["count"] == 2
    assert all(item["split"] == "train" for item in report["items"])
    assert all((root / item["output_image"]).is_file() for item in report["items"])
    assert all((root / item["output_label"]).is_file() for item in report["items"])
    assert all(item["objects"] for item in report["items"])

    second_root = _dataset(tmp_path / "second")
    second = generate_composites(second_root, count=2, seed=11, max_synthetic_ratio=1.0)
    assert [item["labels"] for item in report["items"]] == [item["labels"] for item in second["items"]]


def test_zero_count_does_not_change_dataset(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    report = generate_composites(root, count=0, seed=1)
    assert report["count"] == 0


def test_object_loader_does_not_open_same_source_image_once_per_box(tmp_path: Path, monkeypatch) -> None:
    from research import tomato_pumpkin_lemon_composites as composites

    root = _dataset(tmp_path)
    opened = 0
    original = composites.Image.open

    def counted_open(*args, **kwargs):
        nonlocal opened
        opened += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(composites.Image, "open", counted_open)
    objects = composites._read_train_objects(root)

    assert len(objects) == 3
    assert opened == 3


def test_composites_cover_required_scene_conditions(tmp_path: Path) -> None:
    root = _dataset(tmp_path)

    report = generate_composites(root, count=20, seed=11, max_synthetic_ratio=10.0)

    scenarios = {item["scenario"] for item in report["items"]}
    assert {"separated", "adjacent", "touching", "overlap", "small", "dark", "bright", "blur", "noise"} <= scenarios
