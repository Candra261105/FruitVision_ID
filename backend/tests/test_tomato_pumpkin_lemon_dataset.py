from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from research.tomato_pumpkin_lemon_dataset import (
    CLASS_NAMES,
    build_dataset,
    validate_dataset,
)


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for index, class_id in enumerate((0, 1, 2)):
        image_name = f"image_{index}.jpg"
        Image.new("RGB", (80, 60), (20 + index * 40, 40, 80)).save(root / "images" / image_name)
        (root / "labels" / f"image_{index}.txt").write_text(
            f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    return root


def test_build_dataset_keeps_required_class_order_and_valid_labels(tmp_path: Path) -> None:
    source = _source(tmp_path)
    audit = tmp_path / "audit.json"
    records = []
    for index, class_name in enumerate(CLASS_NAMES):
        image = source / "images" / f"image_{index}.jpg"
        records.append(
            {
                "source_id": "source",
                "image": str(image),
                "label": str(source / "labels" / f"image_{index}.txt"),
                "sha256": "hash-" + str(index),
                "phash": f"{index:016x}",
                "boxes": [{"class_id": index, "class_name": class_name, "box": [0.5, 0.5, 0.5, 0.5]}],
                "errors": [],
            }
        )
    audit.write_text(
        json.dumps(
            {
                "class_names": list(CLASS_NAMES),
                "records": records,
                "class_status": {name: {"valid_box_count": 1} for name in CLASS_NAMES},
                "partial": False,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "dataset"
    report = build_dataset(audit, output, seed=7, split_ratios=(0.25, 0.25, 0.25, 0.25))

    assert report["class_names"] == list(CLASS_NAMES)
    assert (output / "data.yaml").is_file()
    assert validate_dataset(output)["invalid_label_count"] == 0
    assert set(report["split_counts"]) == {"train", "val", "test", "external_holdout"}


def test_validation_rejects_cross_split_exact_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test", "external_holdout"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    image = Image.new("RGB", (20, 20), "red")
    for split in ("train", "val"):
        image.save(root / "images" / split / "same.jpg")
        (root / "labels" / split / "same.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    report = validate_dataset(root)

    assert report["cross_split_duplicate_groups"]
