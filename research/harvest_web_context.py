"""Harvest and convert real-world kitchen and web context images to YOLO format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from research.ai_auto_label import generate_yolo_label_lines


def ingest_kitchen_coco_dataset(
    json_path: Path,
    image_base_dir: Path,
    output_image_dir: Path,
    output_label_dir: Path,
) -> list[dict[str, Any]]:
    """Convert kitchen fruit annotations (lemon=7 -> 2, tomato=13 -> 0) to YOLO."""
    records: list[dict[str, Any]] = []
    if not json_path.is_file() or not image_base_dir.exists():
        return records

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Class mapping: kitchen COCO id -> our target class id
    # 7 (lemon) -> 2, 13 (tomato) -> 0
    class_map = {7: 2, 13: 0}

    images_dict = {img["id"]: img for img in data["images"]}
    annotations_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        cid = ann.get("category_id")
        if cid in class_map:
            img_id = ann["image_id"]
            annotations_by_img.setdefault(img_id, []).append(ann)

    for img_id, anns in annotations_by_img.items():
        img_info = images_dict.get(img_id)
        if not img_info:
            continue
        rel_path = img_info["file_name"]
        src_img_path = image_base_dir / rel_path
        if not src_img_path.is_file():
            # Try without leading subdirs
            src_img_path = image_base_dir / Path(rel_path).name
        if not src_img_path.is_file():
            continue

        img_w = float(img_info.get("width", 0))
        img_h = float(img_info.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            with Image.open(src_img_path) as im:
                img_w, img_h = float(im.width), float(im.height)

        yolo_boxes: list[tuple[int, float, float, float, float]] = []
        for ann in anns:
            target_cls = class_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0:
                yolo_boxes.append((target_cls, cx, cy, norm_w, norm_h))

        if not yolo_boxes:
            continue

        dst_img_name = f"kitchen_{src_img_path.name}"
        dst_img_path = output_image_dir / dst_img_name
        shutil.copy2(src_img_path, dst_img_path)

        dst_lbl_name = f"{Path(dst_img_name).stem}.txt"
        dst_lbl_path = output_label_dir / dst_lbl_name
        label_text = generate_yolo_label_lines(yolo_boxes)
        dst_lbl_path.write_text(label_text, encoding="utf-8")

        records.append(
            {
                "image": str(dst_img_path),
                "label": str(dst_lbl_path),
                "box_count": len(yolo_boxes),
            }
        )

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest real kitchen & web contexts.")
    parser.add_argument(
        "--kitchen-json",
        type=Path,
        default=Path("data/raw/kitchen/annotations.json"),
    )
    parser.add_argument(
        "--kitchen-images",
        type=Path,
        default=Path("data/raw/kitchen/data/test"),
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        default=Path("data/processed/web_context/images/train"),
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=Path("data/processed/web_context/labels/train"),
    )
    args = parser.parse_args()

    print("[*] Ingesting kitchen dataset...")
    records = ingest_kitchen_coco_dataset(
        args.kitchen_json,
        args.kitchen_images,
        args.output_images,
        args.output_labels,
    )
    print(f"[OK] Ingested {len(records)} kitchen scene images with YOLO annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
