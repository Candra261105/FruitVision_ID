from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root: Path) -> dict[str, Any]:
    hashes: dict[str, list[str]] = defaultdict(list)
    split_counts: dict[str, int] = defaultdict(int)
    invalid_labels: list[str] = []
    for image in sorted(path for path in (root / "images").rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        relative = image.relative_to(root / "images")
        split = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        split_counts[split] += 1
        digest = sha256_file(image)
        hashes[digest].append(f"{split}:{image}")
        label = root / "labels" / relative.with_suffix(".txt")
        if not label.is_file():
            invalid_labels.append(f"missing_label:{image}")
            continue
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5:
                invalid_labels.append(f"columns:{label}:{line_number}")
                continue
            try:
                class_id, *box = map(float, fields)
            except ValueError:
                invalid_labels.append(f"number:{label}:{line_number}")
                continue
            if class_id != int(class_id) or class_id < 0 or any(value < 0 or value > 1 for value in box) or box[2] <= 0 or box[3] <= 0:
                invalid_labels.append(f"range:{label}:{line_number}")

    duplicate_groups = {digest: locations for digest, locations in hashes.items() if len(locations) > 1}
    split_collisions = {
        digest: locations
        for digest, locations in duplicate_groups.items()
        if len({location.split(":", 1)[0] for location in locations}) > 1
    }
    return {
        "root": str(root),
        "split_counts": dict(sorted(split_counts.items())),
        "image_count": sum(split_counts.values()),
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": split_collisions,
        "invalid_label_count": len(invalid_labels),
        "invalid_labels": invalid_labels[:100],
        "status": "review_required" if invalid_labels or split_collisions else "clean",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit hash, split, dan label YOLO detector.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "clean" else 2


if __name__ == "__main__":
    raise SystemExit(main())
