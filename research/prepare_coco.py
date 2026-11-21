from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from research.audit_dataset import normalize_class_name


def convert_coco_source(source: Path, annotation_path: Path, output: Path, target_names: list[str]) -> dict[str, int]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: normalize_class_name(item["name"]) for item in payload.get("categories", [])}
    images = {item["id"]: item for item in payload.get("images", [])}
    annotations: dict[int, list[dict]] = {}
    for annotation in payload.get("annotations", []):
        annotations.setdefault(annotation.get("image_id"), []).append(annotation)
    target_ids = {name: index for index, name in enumerate(target_names)}
    image_output = output / "images"
    label_output = output / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in target_names}
    for image_id, image_info in images.items():
        source_image = source / str(image_info.get("file_name", ""))
        if not source_image.is_file():
            continue
        width = float(image_info.get("width", 0))
        height = float(image_info.get("height", 0))
        if width <= 0 or height <= 0:
            continue
        lines: list[str] = []
        for annotation in annotations.get(image_id, []):
            name = categories.get(annotation.get("category_id"))
            if name not in target_ids:
                continue
            box = annotation.get("bbox", [])
            if len(box) != 4:
                continue
            x, y, box_width, box_height = (float(value) for value in box)
            if box_width <= 0 or box_height <= 0 or x >= width or y >= height or x + box_width <= 0 or y + box_height <= 0:
                continue
            class_id = target_ids[name]
            lines.append(f"{class_id} {(x + box_width / 2) / width:.8f} {(y + box_height / 2) / height:.8f} {box_width / width:.8f} {box_height / height:.8f}")
            counts[name] += 1
        target_image = image_output / source_image.name
        if target_image.exists():
            target_image = image_output / f"{image_id}_{source_image.name}"
        shutil.copy2(source_image, target_image)
        (label_output / f"{target_image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (output / "classes.json").write_text(json.dumps(target_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images",
                "val": "images",
                "test": "images",
                "names": target_names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {name: count for name, count in counts.items() if count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Konversi COCO ke YOLO untuk kelas target.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", nargs=3, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_coco_source(args.source, args.annotations, args.output, args.target), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
