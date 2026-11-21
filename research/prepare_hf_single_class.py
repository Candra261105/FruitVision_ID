from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def _bytes(value: Any) -> bytes:
    if isinstance(value, dict) and "bytes" in value:
        return bytes(value["bytes"])
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError("kolom image tidak berisi bytes")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _yolo_line(box: list[Any], width: int, height: int) -> str | None:
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    x, y, box_width, box_height = (float(item) for item in box)
    if box_width <= 0 or box_height <= 0:
        return None
    return f"0 {(x + box_width / 2) / width:.8f} {(y + box_height / 2) / height:.8f} {box_width / width:.8f} {box_height / height:.8f}"


def split_parquet_single_class(
    source: Path,
    output: Path,
    class_name: str,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, int]:
    import pyarrow.parquet as pq

    records: list[tuple[str, bytes, str]] = []
    for row in pq.ParquetFile(source).read().to_pylist():
        image_bytes = _bytes(row.get("image"))
        objects = row.get("objects") or {}
        lines: list[str] = []
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
        for box in objects.get("bbox") or []:
            line = _yolo_line(box, width, height)
            if line is not None:
                lines.append(line)
        if lines:
            records.append((_hash_bytes(image_bytes), image_bytes, "\n".join(lines) + "\n"))
    unique: dict[str, tuple[bytes, str]] = {image_hash: (image, label) for image_hash, image, label in records}
    ordered = sorted(unique.items())
    random.Random(seed).shuffle(ordered)
    holdout_count = max(1, int(round(len(ordered) * holdout_fraction))) if ordered else 0
    holdout = ordered[:holdout_count]
    train = ordered[holdout_count:]
    for split, items in (("train", train), ("holdout", holdout)):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, (image_hash, (image_bytes, label)) in enumerate(items):
            name = f"{class_name}_{index:04d}_{image_hash[:10]}"
            (image_dir / f"{name}.jpg").write_bytes(image_bytes)
            (label_dir / f"{name}.txt").write_text(label, encoding="utf-8")
    (output / "classes.json").write_text(json.dumps([class_name], indent=2), encoding="utf-8")
    return {"source_images": len(records), "unique_images": len(ordered), "train_images": len(train), "holdout_images": len(holdout), "boxes": sum(len(label.splitlines()) for _, (_, label) in ordered)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Siapkan dataset Parquet satu kelas dengan holdout lintas sumber.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(split_parquet_single_class(args.source, args.output, args.class_name, args.holdout_fraction, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
