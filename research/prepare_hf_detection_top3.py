from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

TARGET_NAMES = ("cucumber", "tomato", "green beans")
TARGET_SOURCE_NAMES = {
    "cucumber": "cucumber",
    "green beans": "green beans",
    "tomato": "tomato",
}


def _bytes(value: Any) -> bytes:
    if isinstance(value, dict) and "bytes" in value:
        return bytes(value["bytes"])
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError("kolom image tidak berisi bytes")


def _box_line(class_id: int, box: list[Any], width: int, height: int) -> str | None:
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    x, y, box_width, box_height = (float(value) for value in box)
    if box_width <= 0 or box_height <= 0:
        return None
    if x >= width or y >= height or x + box_width <= 0 or y + box_height <= 0:
        return None
    return f"{class_id} {(x + box_width / 2) / width:.8f} {(y + box_height / 2) / height:.8f} {box_width / width:.8f} {box_height / height:.8f}"


def convert_parquet_top3(source: Path, output: Path, class_names: dict[int, str], dataset_id: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    normalized = {class_id: str(name).strip().lower() for class_id, name in class_names.items()}
    target_ids = {name: index for index, name in enumerate(TARGET_NAMES)}
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    rows_seen = 0
    skipped = Counter()
    for row_index, row in enumerate(pq.ParquetFile(source).iter_batches(batch_size=64)):
        for offset, record in enumerate(row.to_pylist()):
            rows_seen += 1
            split = str(record.get("split") or "train").lower()
            if split not in {"train", "validation", "test"}:
                skipped["unknown_split"] += 1
                continue
            objects = record.get("objects") or {}
            boxes = objects.get("bbox") or []
            categories = objects.get("categories") or []
            image_bytes = _bytes(record.get("image"))
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                width, height = decoded.size
            lines: list[str] = []
            for box, category in zip(boxes, categories):
                source_name = normalized.get(int(category), "")
                target_name = TARGET_SOURCE_NAMES.get(source_name)
                if target_name is None:
                    continue
                line = _box_line(target_ids[target_name], box, width, height)
                if line is None:
                    skipped["invalid_box"] += 1
                    continue
                lines.append(line)
                per_split[split][target_name] += 1
            if not lines:
                skipped["no_target"] += 1
                continue
            image_name = f"{dataset_id}_{row_index:06d}_{offset:03d}.jpg"
            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / image_name).write_bytes(image_bytes)
            (label_dir / image_name.replace(".jpg", ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    (output / "classes.json").write_text(json.dumps(list(TARGET_NAMES), ensure_ascii=False, indent=2), encoding="utf-8")
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": list(TARGET_NAMES),
    }
    import yaml

    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {"rows_seen": rows_seen, "per_split": {key: dict(value) for key, value in per_split.items()}, "skipped": dict(skipped)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Konversi HF Parquet smartphone vegetables ke tiga kelas FruitVision.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    args = parser.parse_args()
    classes = {int(key): str(value) for key, value in json.loads(args.classes.read_text(encoding="utf-8-sig")).items()}
    print(json.dumps(convert_parquet_top3(args.source, args.output, classes, "smartphone"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
