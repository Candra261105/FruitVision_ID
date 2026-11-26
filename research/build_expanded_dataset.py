"""Integrate real images, frozen benchmark splits, and synthetic composites into unified expanded dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = ("tomato", "pumpkin", "lemon")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_benchmark_splits(source_dataset: Path, target_dataset: Path) -> dict[str, int]:
    """Copy benchmark val and test splits directly to preserve exact evaluation baseline."""
    counts = {"val_count": 0, "test_count": 0}
    for split in ("val", "test"):
        src_img_dir = source_dataset / "images" / split
        src_lbl_dir = source_dataset / "labels" / split
        tgt_img_dir = target_dataset / "images" / split
        tgt_lbl_dir = target_dataset / "labels" / split

        tgt_img_dir.mkdir(parents=True, exist_ok=True)
        tgt_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            continue

        for img_path in src_img_dir.glob("*"):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            dst_img = tgt_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if lbl_path.is_file():
                dst_lbl = tgt_lbl_dir / lbl_path.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_path, dst_lbl)
            counts[f"{split}_count"] += 1

    return counts


def integrate_train_data(
    source_dataset: Path,
    composite_dir: Path,
    target_dataset: Path,
) -> dict[str, int]:
    """Integrate real train images and newly generated synthetic composites into train split."""
    tgt_img_dir = target_dataset / "images" / "train"
    tgt_lbl_dir = target_dataset / "labels" / "train"
    tgt_img_dir.mkdir(parents=True, exist_ok=True)
    tgt_lbl_dir.mkdir(parents=True, exist_ok=True)

    counts = {"real_train": 0, "synthetic_train": 0}

    # 1. Real train images from base dataset
    src_train_img = source_dataset / "images" / "train"
    src_train_lbl = source_dataset / "labels" / "train"
    if src_train_img.exists():
        for img_path in src_train_img.glob("*"):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # Skip old synthetic if we are replacing with high-grade composites
            if img_path.name.startswith("synthetic_"):
                continue
            lbl_path = src_train_lbl / f"{img_path.stem}.txt"
            dst_img = tgt_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if lbl_path.is_file():
                dst_lbl = tgt_lbl_dir / lbl_path.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_path, dst_lbl)
            counts["real_train"] += 1

    # 2. Synthetic composites
    comp_img_dir = composite_dir / "images" / "train"
    comp_lbl_dir = composite_dir / "labels" / "train"
    if comp_img_dir.exists():
        for img_path in comp_img_dir.glob("*.jpg"):
            lbl_path = comp_lbl_dir / f"{img_path.stem}.txt"
            dst_img = tgt_img_dir / img_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if lbl_path.is_file():
                dst_lbl = tgt_lbl_dir / lbl_path.name
                if not dst_lbl.exists():
                    shutil.copy2(lbl_path, dst_lbl)
            counts["synthetic_train"] += 1

    return counts


def verify_dataset_leakage(target_dataset: Path) -> bool:
    """Verify strictly zero synthetic leakage into val/test, and valid bounding box math."""
    for split in ("val", "test"):
        img_dir = target_dataset / "images" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*"):
            if "synthetic" in img_path.name.lower():
                print(f"[LEAKAGE DETECTED] Synthetic image found in {split}: {img_path.name}")
                return False

    # Check bounding box validity across all splits
    for split in ("train", "val", "test"):
        lbl_dir = target_dataset / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_path in lbl_dir.glob("*.txt"):
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    print(f"[INVALID LABEL] Line format in {lbl_path}: {line}")
                    return False
                cid = int(parts[0])
                if cid not in range(len(CLASS_NAMES)):
                    print(f"[INVALID CLASS ID] Class ID {cid} in {lbl_path}")
                    return False
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    print(f"[OUT OF BOUNDS] Box {parts[1:]} in {lbl_path}")
                    return False

    return True


def write_dataset_yaml(target_dataset: Path) -> Path:
    """Write standardized data.yaml pointing to relative train, val, and test splits."""
    content = {
        "path": "dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": list(CLASS_NAMES),
    }
    yaml_path = target_dataset / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(content, f, sort_keys=False)
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified expanded dataset.")
    parser.add_argument(
        "--source-benchmark",
        type=Path,
        default=Path("data/tomato-pumpkin-lemon-training/dataset"),
    )
    parser.add_argument(
        "--composite-dir",
        type=Path,
        default=Path("data/processed/tomato-pumpkin-lemon-expanded"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tomato-pumpkin-lemon-expanded/dataset"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("[*] Freezing benchmark val & test splits...")
    freeze_stats = freeze_benchmark_splits(args.source_benchmark, args.output)
    print(f"  -> Frozen val: {freeze_stats['val_count']}, test: {freeze_stats['test_count']}")

    print("[*] Integrating real train images and synthetic composites...")
    train_stats = integrate_train_data(args.source_benchmark, args.composite_dir, args.output)
    print(f"  -> Real train images: {train_stats['real_train']}")
    print(f"  -> Synthetic composites: {train_stats['synthetic_train']}")

    print("[*] Verifying data integrity and leakage guard...")
    is_valid = verify_dataset_leakage(args.output)
    if not is_valid:
        print("[ERROR] Dataset verification failed! Check leakage or coordinate anomalies.")
        return 1

    print("[*] Writing data.yaml...")
    yaml_path = write_dataset_yaml(args.output)

    # Compute object statistics
    object_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for lbl_file in (args.output / "labels" / "train").glob("*.txt"):
        for line in lbl_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cid = int(parts[0])
                if cid in range(len(CLASS_NAMES)):
                    object_counts[CLASS_NAMES[cid]] += 1

    report = {
        "dataset_name": "tomato-pumpkin-lemon-expanded",
        "splits": {
            "train_real": train_stats["real_train"],
            "train_synthetic": train_stats["synthetic_train"],
            "train_total": train_stats["real_train"] + train_stats["synthetic_train"],
            "val": freeze_stats["val_count"],
            "test": freeze_stats["test_count"],
        },
        "train_objects_per_class": object_counts,
        "classes": list(CLASS_NAMES),
        "data_yaml": str(yaml_path),
    }

    report_path = args.output / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[OK] Dataset expansion successfully assembled!")
    print(f"Total train images: {report['splits']['train_total']}")
    print(f"Val images (frozen): {report['splits']['val']}")
    print(f"Test images (frozen): {report['splits']['test']}")
    print("Objects per class in train:")
    for cname, count in object_counts.items():
        print(f"  * {cname}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
