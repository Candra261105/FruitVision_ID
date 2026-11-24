"""Create conservative train-only composites from verified train images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageChops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tomato_pumpkin_lemon_dataset import CLASS_NAMES


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_train_objects(root: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for image in sorted((root / "images" / "train").glob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        label = root / "labels" / "train" / f"{image.stem}.txt"
        if not label.is_file():
            continue
        parsed_boxes: list[tuple[int, float, float, float, float]] = []
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            class_id = int(fields[0])
            if class_id not in range(len(CLASS_NAMES)):
                continue
            parsed_boxes.append((class_id, *(float(value) for value in fields[1:])))
        if not parsed_boxes:
            continue
        with Image.open(image) as source:
            for class_id, cx, cy, width, height in parsed_boxes:
                image_width, image_height = source.size
                left = max(0, int((cx - width / 2) * image_width))
                top = max(0, int((cy - height / 2) * image_height))
                right = min(image_width, int((cx + width / 2) * image_width))
                bottom = min(image_height, int((cy + height / 2) * image_height))
                if right - left < 8 or bottom - top < 8:
                    continue
                crop = source.convert("RGB").crop((left, top, right, bottom))
                objects.append(
                    {
                        "source_image": str(image),
                        "source_sha256": _hash(image),
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "crop": crop,
                    }
                )
    return objects


def generate_composites(
    dataset_root: Path,
    count: int,
    seed: int,
    max_synthetic_ratio: float = 1.0,
) -> dict[str, Any]:
    if count < 0:
        raise ValueError("count tidak boleh negatif")
    if not 0 <= max_synthetic_ratio:
        raise ValueError("max_synthetic_ratio tidak valid")
    if count == 0:
        report = {"schema_version": 1, "seed": seed, "count": 0, "items": [], "max_synthetic_ratio": max_synthetic_ratio}
        (dataset_root / "synthetic_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    objects = _read_train_objects(dataset_root)
    if len(objects) < 3:
        raise ValueError("minimal tiga object train valid diperlukan")
    real_train_count = len(list((dataset_root / "images" / "train").glob("*")))
    allowed = int(real_train_count * max_synthetic_ratio)
    count = min(count, allowed)
    rng = random.Random(seed)
    output_items: list[dict[str, Any]] = []
    out_image_dir = dataset_root / "images" / "train"
    out_label_dir = dataset_root / "labels" / "train"
    for index in range(count):
        canvas = Image.new("RGB", (640, 480), (230, 230, 230))
        labels: list[str] = []
        metadata: list[dict[str, Any]] = []
        scenarios = ("separated", "adjacent", "touching", "overlap", "small", "dark", "bright", "blur", "noise")
        scenario = scenarios[index % len(scenarios)]
        chosen = [objects[(index * 3 + offset) % len(objects)] for offset in range(3)]
        positions_by_scenario = {
            "separated": ((30, 40), (220, 110), (410, 230)),
            "adjacent": ((50, 120), (220, 120), (390, 120)),
            "touching": ((50, 120), (190, 120), (330, 120)),
            "overlap": ((70, 100), (155, 140), (300, 170)),
            "small": ((150, 80), (280, 180), (410, 280)),
            "dark": ((30, 40), (220, 110), (410, 230)),
            "bright": ((30, 40), (220, 110), (410, 230)),
            "blur": ((30, 40), (220, 110), (410, 230)),
            "noise": ((30, 40), (220, 110), (410, 230)),
        }
        positions = positions_by_scenario[scenario]
        for object_index, (obj, position) in enumerate(zip(chosen, positions)):
            crop = obj["crop"].copy()
            scale = 0.28 if scenario == "small" else rng.uniform(0.55, 1.05)
            crop.thumbnail((int(180 * scale), int(180 * scale)), Image.Resampling.LANCZOS)
            if scenario == "blur" or rng.random() < 0.35:
                crop = crop.filter(ImageFilter.GaussianBlur(radius=0.7))
            if scenario == "dark":
                crop = ImageEnhance.Brightness(crop).enhance(0.45)
            elif scenario == "bright":
                crop = ImageEnhance.Brightness(crop).enhance(1.5)
            elif rng.random() < 0.25:
                crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.65, 1.25))
            if scenario == "noise":
                noise = Image.effect_noise(crop.size, 28).convert("RGB")
                crop = ImageChops.blend(crop, noise, 0.18)
            x, y = position
            canvas.paste(crop, (x, y))
            cx = (x + crop.width / 2) / canvas.width
            cy = (y + crop.height / 2) / canvas.height
            width = crop.width / canvas.width
            height = crop.height / canvas.height
            labels.append(f"{obj['class_id']} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
            metadata.append(
                {
                    "class_id": obj["class_id"],
                    "class_name": obj["class_name"],
                    "source_image": obj["source_image"],
                    "source_sha256": obj["source_sha256"],
                    "position": [x, y],
                    "size": [crop.width, crop.height],
                }
            )
        stem = f"synthetic_{seed}_{index:05d}"
        image_path = out_image_dir / f"{stem}.jpg"
        label_path = out_label_dir / f"{stem}.txt"
        canvas.save(image_path, quality=92)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        output_items.append(
            {
                "output_image": str(image_path.relative_to(dataset_root)),
                "output_label": str(label_path.relative_to(dataset_root)),
                "split": "train",
                "seed": seed + index,
                "scenario": scenario,
                "synthetic": True,
                "objects": metadata,
                "labels": labels,
                "sha256": _hash(image_path),
            }
        )
    report = {
        "schema_version": 1,
        "seed": seed,
        "count": len(output_items),
        "max_synthetic_ratio": max_synthetic_ratio,
        "items": output_items,
    }
    (dataset_root / "synthetic_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_report_path = dataset_root / "dataset_report.json"
    if dataset_report_path.is_file():
        dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
        dataset_report["synthetic_images"] = len(output_items)
        dataset_report["split_counts"]["train"] = int(dataset_report["split_counts"].get("train", 0)) + len(output_items)
        dataset_report_path.write_text(json.dumps(dataset_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Buat komposit train-only tomato/pumpkin/lemon.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--max-synthetic-ratio", type=float, default=0.5)
    args = parser.parse_args()
    report = generate_composites(args.dataset, args.count, args.seed, args.max_synthetic_ratio)
    print(json.dumps({"count": report["count"], "dataset": str(args.dataset)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
