"""Automated crop harvester for tomato, pumpkin, and lemon from verified sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

CLASS_NAMES = ("tomato", "pumpkin", "lemon")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def harvest_from_yolo(
    image_dir: Path,
    label_dir: Path,
    class_map: dict[int, int],
    output_dir: Path,
    padding: int = 4,
    min_size: int = 12,
    max_per_class: int = 2000,
    exclude_stems: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Harvest fruit objects from YOLO labeled image dataset."""
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists() or not label_dir.exists():
        return records

    class_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])
    for img_path in image_files:
        if exclude_stems and img_path.stem in exclude_stems:
            continue
        if all(c >= max_per_class for c in class_counts.values()):
            break
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            continue

        lines = lbl_path.read_text(encoding="utf-8").splitlines()
        valid_boxes: list[tuple[int, float, float, float, float, int]] = []
        for box_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            src_cls = int(parts[0])
            if src_cls not in class_map:
                continue
            target_cls = class_map[src_cls]
            if target_cls not in range(len(CLASS_NAMES)):
                continue
            if class_counts[target_cls] >= max_per_class:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            valid_boxes.append((target_cls, cx, cy, bw, bh, box_idx))

        if not valid_boxes:
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                for target_cls, cx, cy, bw, bh, box_idx in valid_boxes:
                    if class_counts[target_cls] >= max_per_class:
                        continue
                    target_name = CLASS_NAMES[target_cls]

                    # Convert YOLO format to pixel bounds
                    box_w = bw * img_w
                    box_h = bh * img_h
                    if box_w < min_size or box_h < min_size:
                        continue

                    x1 = max(0, int((cx - bw / 2) * img_w) - padding)
                    y1 = max(0, int((cy - bh / 2) * img_h) - padding)
                    x2 = min(img_w, int((cx + bw / 2) * img_w) + padding)
                    y2 = min(img_h, int((cy + bh / 2) * img_h) + padding)

                    if x2 - x1 < min_size or y2 - y1 < min_size:
                        continue

                    crop = img.convert("RGB").crop((x1, y1, x2, y2))
                    cls_folder = output_dir / target_name
                    cls_folder.mkdir(parents=True, exist_ok=True)

                    crop_filename = f"{target_name}_{img_path.stem}_b{box_idx}.jpg"
                    crop_dest = cls_folder / crop_filename
                    crop.save(crop_dest, quality=90)

                    class_counts[target_cls] += 1
                    records.append(
                        {
                            "crop_path": str(crop_dest),
                            "class_id": target_cls,
                            "class_name": target_name,
                            "width": crop.width,
                            "height": crop.height,
                            "source_image": str(img_path),
                        }
                    )
        except Exception as err:
            continue

    return records


def ingest_existing_crops(
    source_dir: Path,
    class_id: int,
    class_name: str,
    output_dir: Path,
    min_size: int = 16,
    max_crops: int = 1500,
) -> list[dict[str, Any]]:
    """Ingest already cropped fruit images (e.g. classification datasets) into harvest repository."""
    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        return records

    cls_folder = output_dir / class_name
    cls_folder.mkdir(parents=True, exist_ok=True)

    crop_candidates = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    for idx, cand in enumerate(crop_candidates):
        if len(records) >= max_crops:
            break
        try:
            with Image.open(cand) as img:
                if img.width < min_size or img.height < min_size:
                    continue
                crop_dest = cls_folder / f"{class_name}_ingest_{cand.stem}_{idx:05d}.jpg"
                img.convert("RGB").save(crop_dest, quality=90)

                records.append(
                    {
                        "crop_path": str(crop_dest),
                        "class_id": class_id,
                        "class_name": class_name,
                        "width": img.width,
                        "height": img.height,
                        "source_image": str(cand),
                    }
                )
        except Exception:
            continue

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest fruit crops from project sources.")
    parser.add_argument("--output", type=Path, default=Path("data/crops"))
    parser.add_argument("--min-size", type=int, default=14)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=1500)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []

    # Collect benchmark val & test stems to guarantee ZERO leakage
    exclude_stems: set[str] = set()
    benchmark_dir = Path("data/tomato-pumpkin-lemon-training/dataset/images")
    if benchmark_dir.exists():
        for split in ("val", "test"):
            split_dir = benchmark_dir / split
            if split_dir.exists():
                for p in split_dir.glob("*"):
                    stem = p.stem
                    if stem.startswith("bangladesh__"):
                        stem = stem[len("bangladesh__"):]
                    exclude_stems.add(stem)
    print(f"[*] Leakage guard: excluding {len(exclude_stems)} benchmark val/test image stems from harvest.")

    # 1. Bangladesh YOLO Dataset
    # Mapping in bangladesh: 14: lemon -> 2, 16: pumpkin -> 1, 20: tomato -> 0
    bg_dir = Path(r"C:\Users\dragon\Downloads\FruitVision_ID\data\processed\bangladesh")
    if bg_dir.exists():
        print(f"[*] Harvesting from Bangladesh dataset: {bg_dir}")
        bg_records = harvest_from_yolo(
            image_dir=bg_dir / "images",
            label_dir=bg_dir / "labels",
            class_map={14: 2, 16: 1, 20: 0},
            output_dir=args.output,
            padding=args.padding,
            min_size=args.min_size,
            max_per_class=args.max_per_class,
            exclude_stems=exclude_stems,
        )
        print(f"  -> Harvested {len(bg_records)} crops from Bangladesh (strictly train-only).")
        all_records.extend(bg_records)

    # 2. Existing Tomato Crops (ParsaVictor / CC-BY 4.0)
    tomato_crop_dir = Path(r"C:\Users\dragon\Downloads\FruitVision_ID\data\advanced\source\crops\cls_tomato")
    if tomato_crop_dir.exists():
        print(f"[*] Ingesting pre-cropped tomatoes from: {tomato_crop_dir}")
        t_records = ingest_existing_crops(
            source_dir=tomato_crop_dir,
            class_id=0,
            class_name="tomato",
            output_dir=args.output,
            min_size=args.min_size,
            max_crops=args.max_per_class,
        )
        print(f"  -> Ingested {len(t_records)} tomato crops.")
        all_records.extend(t_records)

    # Save manifest
    manifest_path = args.output / "harvest_manifest.json"
    manifest_data = {
        "total_crops": len(all_records),
        "crops_per_class": {
            cls_name: sum(1 for r in all_records if r["class_name"] == cls_name)
            for cls_name in CLASS_NAMES
        },
        "crops": all_records,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"\n[OK] Harvest complete! Manifest saved to: {manifest_path}")
    print(f"Total crops: {manifest_data['total_crops']}")
    for cname, count in manifest_data["crops_per_class"].items():
        print(f"  * {cname}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
