"""Audit sumber YOLO untuk dataset tomato/pumpkin/lemon.

Modul ini tidak menebak kelas dari nama file atau warna piksel. Setiap sumber
harus memberikan pemetaan class-id yang eksplisit. Kelas lain pada sumber
diabaikan, bukan direlabel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

CLASS_NAMES = ("tomato", "pumpkin", "lemon")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Pemetaan yang diverifikasi dari data/processed/bangladesh/data.yaml.
BANGLADESH_CLASS_MAP = {14: "lemon", 16: "pumpkin", 20: "tomato"}
SOURCE_CLASS_COUNTS = {"bangladesh": 22}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    """Return a small average hash for near-duplicate grouping."""
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
            flattened = getattr(gray, "get_flattened_data", None)
            pixels = list(flattened()) if callable(flattened) else list(gray.getdata())
    except (OSError, ValueError):
        return "invalid-image"
    average = sum(pixels) / len(pixels)
    bits = sum((1 << index) for index, value in enumerate(pixels) if value >= average)
    return f"{bits:016x}"


def validate_yolo_label(line: str, class_names: Iterable[str] = CLASS_NAMES) -> list[str]:
    """Return validation errors for one normalized YOLO label line."""
    fields = line.split()
    if len(fields) != 5:
        return ["column_count"]
    errors: list[str] = []
    try:
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
    except ValueError:
        return ["not_numeric"]
    if class_id < 0 or class_id >= len(tuple(class_names)):
        errors.append("class_id_out_of_range")
    if not all(0.0 <= value <= 1.0 for value in values):
        errors.append("box_out_of_range")
    if values[2] <= 0.0 or values[3] <= 0.0:
        errors.append("box_zero_area")
    return errors


def _image_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _label_for_image(source: Path, image: Path) -> Path:
    try:
        relative = image.relative_to(source / "images")
        return (source / "labels" / relative).with_suffix(".txt")
    except ValueError:
        return image.with_suffix(".txt")


def _source_id(source: Path) -> str:
    return source.name.lower().replace(" ", "-") or "source"


def audit_sources(
    sources: list[Path],
    output: Path,
    class_maps: dict[str, dict[int, str]] | None = None,
) -> dict[str, Any]:
    """Audit YOLO sources and write a deterministic JSON report.

    ``class_maps`` is keyed by source directory name. If omitted, the known
    Bangladesh source mapping is used for a directory named ``bangladesh``.
    """
    class_maps = class_maps or {}
    records: list[dict[str, Any]] = []
    class_status = {
        name: {"valid_box_count": 0, "valid_image_count": 0, "sources": []}
        for name in CLASS_NAMES
    }

    for source in sorted(path.resolve() for path in sources):
        source_id = _source_id(source)
        mapping = class_maps.get(source_id)
        if mapping is None and source_id == "bangladesh":
            mapping = BANGLADESH_CLASS_MAP
        mapping = mapping or {}
        for image in _image_files(source):
            label = _label_for_image(source, image)
            errors: list[str] = []
            mapped_boxes: list[dict[str, Any]] = []
            counts: Counter[str] = Counter()
            if not label.is_file():
                errors.append("missing_label")
            else:
                for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    source_class_count = SOURCE_CLASS_COUNTS.get(source_id, max(mapping, default=-1) + 1)
                    line_errors = validate_yolo_label(line, tuple(str(i) for i in range(source_class_count)))
                    if line_errors:
                        errors.extend(f"{error}:line_{line_number}" for error in line_errors)
                        continue
                    class_id = int(fields[0])
                    class_name = mapping.get(class_id)
                    if class_name not in CLASS_NAMES:
                        continue
                    box = [float(value) for value in fields[1:]]
                    mapped_boxes.append({"class_id": CLASS_NAMES.index(class_name), "class_name": class_name, "box": box})
                    counts[class_name] += 1

            record = {
                "source_id": source_id,
                "image": str(image),
                "label": str(label),
                "sha256": sha256_file(image),
                "phash": perceptual_hash(image),
                "class_counts": dict(sorted(counts.items())),
                "boxes": mapped_boxes,
                "errors": errors,
            }
            if record["phash"] == "invalid-image" and label.is_file():
                record["errors"].append("invalid_image")
            records.append(record)
            for class_name, count in counts.items():
                status = class_status[class_name]
                status["valid_box_count"] += count
                status["sources"] = sorted(set(status["sources"]) | {source_id})
                status["valid_image_count"] += 1

    invalid_records = [record for record in records if record["errors"]]
    report: dict[str, Any] = {
        "schema_version": 1,
        "class_names": list(CLASS_NAMES),
        "sources": [str(path.resolve()) for path in sorted(sources)],
        "class_status": class_status,
        "records": records,
        "record_count": len(records),
        "invalid_record_count": len(invalid_records),
        "partial": any(status["valid_box_count"] == 0 for status in class_status.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit sumber YOLO tomato/pumpkin/lemon.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_sources(args.source, args.output)
    print(json.dumps({"class_status": report["class_status"], "partial": report["partial"]}, indent=2))
    return 0 if report["invalid_record_count"] == 0 and not report["partial"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
