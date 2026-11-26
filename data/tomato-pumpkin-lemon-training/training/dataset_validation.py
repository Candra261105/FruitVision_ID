"""Build and validate the separate tomato/pumpkin/lemon YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = ("tomato", "pumpkin", "lemon")
SPLITS = ("train", "val", "test", "external_holdout")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_groups(records: list[dict[str, Any]], seed: int, ratios: tuple[float, ...]) -> dict[str, str]:
    """Assign records by phash group, with deterministic approximate ratios."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("phash") or record["sha256"])].append(record)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    total = len(records)
    if len(ratios) == 3:
        ratios = (*ratios, 0.0)
    if len(ratios) != 4 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split_ratios harus berisi empat angka dengan jumlah 1")
    targets = {split: total * ratio for split, ratio in zip(SPLITS, ratios)}
    counts = Counter[str]()
    assignment: dict[str, str] = {}
    for group in ordered:
        split = min(SPLITS, key=lambda name: counts[name] - targets[name])
        for record in group:
            assignment[record["sha256"]] = split
            counts[split] += 1
    return assignment


def _write_label(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}".format(
            class_id=int(box["class_id"]),
            cx=float(box["box"][0]),
            cy=float(box["box"][1]),
            width=float(box["box"][2]),
            height=float(box["box"][3]),
        )
        for box in record["boxes"]
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_dataset(
    audit_path: Path,
    output: Path,
    seed: int = 20261124,
    split_ratios: tuple[float, ...] = (0.70, 0.10, 0.10, 0.10),
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if tuple(audit.get("class_names", ())) != CLASS_NAMES:
        raise ValueError(f"audit class order must be {CLASS_NAMES}")
    records = [
        record
        for record in audit.get("records", [])
        if not record.get("errors") and record.get("boxes") and Path(record["image"]).is_file()
    ]
    if not records:
        raise ValueError("audit tidak memiliki record target yang valid")
    output.mkdir(parents=True, exist_ok=True)
    assignment = _split_groups(records, seed, split_ratios)
    split_counts = Counter[str]()
    object_counts: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    used_hashes: dict[str, str] = {}
    for record in records:
        split = assignment[record["sha256"]]
        image = Path(record["image"])
        destination_name = f"{record['source_id']}__{image.stem}{image.suffix.lower()}"
        destination = output / "images" / split / destination_name
        if record["sha256"] in used_hashes:
            raise ValueError(f"duplicate image assigned twice: {record['sha256']}")
        used_hashes[record["sha256"]] = split
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
        label_destination = output / "labels" / split / destination.with_suffix(".txt").name
        _write_label(label_destination, record)
        split_counts[split] += 1
        for box in record["boxes"]:
            object_counts[box["class_name"]] += 1
        manifest.append(
            {
                "output_image": str(destination.relative_to(output)),
                "output_label": str(label_destination.relative_to(output)),
                "split": split,
                "source_id": record["source_id"],
                "source_image": record["image"],
                "sha256": record["sha256"],
                "phash": record.get("phash"),
                "synthetic": False,
            }
        )

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(CLASS_NAMES),
    }
    external_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/external_holdout",
        "names": list(CLASS_NAMES),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (output / "data-external.yaml").write_text(yaml.safe_dump(external_yaml, sort_keys=False), encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate_dataset(output)
    report = {
        "schema_version": 1,
        "class_names": list(CLASS_NAMES),
        "seed": seed,
        "source_audit": str(audit_path),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "object_counts": {name: object_counts[name] for name in CLASS_NAMES},
        "synthetic_images": validation["synthetic_images"],
        "synthetic_validation_errors": validation["synthetic_errors"],
        "partial": any(object_counts[name] == 0 for name in CLASS_NAMES) or validation["status"] != "clean",
    }
    (output / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Tomato Pumpkin Lemon\n\n"
        "Dataset eksperimen terpisah dengan urutan kelas `0 tomato`, `1 pumpkin`, `2 lemon`.\n\n"
        "Semua gambar pada tahap ini berasal dari sumber beranotasi asli. Folder `external_holdout` "
        "dipakai sebagai evaluasi terpisah. Data sintetis, bila ditambahkan, hanya boleh masuk `train`.\n",
        encoding="utf-8",
    )
    return report


def validate_dataset(root: Path) -> dict[str, Any]:
    hashes: dict[str, list[str]] = defaultdict(list)
    invalid: list[str] = []
    class_counts: Counter[str] = Counter()
    for split in SPLITS:
        image_dir = root / "images" / split
        for image in sorted(image_dir.glob("*")) if image_dir.is_dir() else []:
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = _sha256(image)
            hashes[digest].append(split)
            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                invalid.append(f"missing_label:{image}")
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    invalid.append(f"columns:{label}:{line_number}")
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    invalid.append(f"number:{label}:{line_number}")
                    continue
                if class_id not in range(len(CLASS_NAMES)):
                    invalid.append(f"class_id:{label}:{line_number}")
                if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                    invalid.append(f"range:{label}:{line_number}")
                if class_id in range(len(CLASS_NAMES)):
                    class_counts[CLASS_NAMES[class_id]] += 1
        label_dir = root / "labels" / split
        image_stems = {
            image.stem
            for image in (root / "images" / split).glob("*")
            if image.suffix.lower() in IMAGE_SUFFIXES
        } if (root / "images" / split).is_dir() else set()
        if label_dir.is_dir():
            for label in sorted(label_dir.glob("*.txt")):
                if label.stem not in image_stems:
                    invalid.append(f"orphan_label:{label}")
    duplicate_groups = {digest: locations for digest, locations in hashes.items() if len(locations) > 1}
    cross_split = {digest: locations for digest, locations in duplicate_groups.items() if len(set(locations)) > 1}
    synthetic_images = 0
    synthetic_errors: list[str] = []
    manifest_path = root / "synthetic_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("items", []):
            image = root / str(item.get("output_image", "")).replace("\\", "/")
            label = root / str(item.get("output_label", "")).replace("\\", "/")
            if item.get("split") != "train":
                synthetic_errors.append(f"synthetic_not_train:{item.get('output_image')}")
            if not image.is_file() or not label.is_file():
                synthetic_errors.append(f"synthetic_missing_pair:{item.get('output_image')}")
            else:
                synthetic_images += 1
    invalid.extend(synthetic_errors)
    return {
        "invalid_label_count": len(invalid),
        "invalid_labels": invalid[:100],
        "class_counts": dict(class_counts),
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": cross_split,
        "synthetic_images": synthetic_images,
        "synthetic_errors": synthetic_errors,
        "status": "clean" if not invalid and not cross_split else "review_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset YOLO tomato/pumpkin/lemon.")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    report = build_dataset(args.audit, args.output, args.seed)
    validation = validate_dataset(args.output)
    print(json.dumps({"report": report, "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "clean" else 2


if __name__ == "__main__":
    raise SystemExit(main())
