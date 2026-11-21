from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate_label_file(path: Path, class_count: int = 5) -> list[str]:
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return ["file bukan UTF-8"]
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        prefix = f"baris {line_number}"
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"{prefix}: harus memiliki 5 field")
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{prefix}: field harus numerik")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{prefix}: class id di luar 0..{class_count - 1}")
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{prefix}: koordinat harus finite")
        elif not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"{prefix}: koordinat harus berada pada 0..1")
        elif values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{prefix}: lebar dan tinggi box harus positif")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi pasangan gambar dan label YOLO.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Folder sumber tidak ditemukan: {args.source}")
    images = sorted(path for path in args.source.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    missing = [str(path.relative_to(args.source)) for path in images if not path.with_suffix(".txt").exists()]
    invalid = {
        str(path.relative_to(args.source)): validate_label_file(path.with_suffix(".txt"))
        for path in images
        if path.with_suffix(".txt").exists() and validate_label_file(path.with_suffix(".txt"))
    }
    report = {
        "source": str(args.source.resolve()),
        "image_count": len(images),
        "missing_label_count": len(missing),
        "missing_labels": missing[:1000],
        "invalid_label_count": len(invalid),
        "invalid_labels": invalid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not missing and not invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
