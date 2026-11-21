from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from research.audit_dataset import normalize_class_name


def remap_yolo_text(label_text: str, source_names: list[str], target_names: list[str]) -> str:
    target_ids = {name: index for index, name in enumerate(target_names)}
    lines: list[str] = []
    for raw_line in label_text.splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        try:
            local_id = int(fields[0])
        except ValueError:
            continue
        if not 0 <= local_id < len(source_names):
            continue
        target_id = target_ids.get(normalize_class_name(source_names[local_id]))
        if target_id is None or len(fields) != 5:
            continue
        lines.append(" ".join([str(target_id), *fields[1:]]))
    return "\n".join(lines) + ("\n" if lines else "")


def load_source_names(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(value) for value in payload]
    if isinstance(payload, dict) and "names" in payload:
        names = payload["names"]
        if isinstance(names, list):
            return [str(value) for value in names]
        if isinstance(names, dict):
            return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    raise ValueError(f"format nama kelas tidak dikenali: {path}")


def find_parallel_label(source: Path, image: Path) -> Path | None:
    beside = image.with_suffix(".txt")
    if beside.is_file():
        return beside
    try:
        relative = image.relative_to(source / "images")
    except ValueError:
        return None
    parallel = (source / "labels" / relative).with_suffix(".txt")
    return parallel if parallel.is_file() else None


def prepare_yolo_source(source: Path, output: Path, source_names: list[str], target_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in target_names}
    image_output = output / "images"
    label_output = output / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    for image in sorted(source.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        label = find_parallel_label(source, image)
        if label is None:
            continue
        remapped = remap_yolo_text(label.read_text(encoding="utf-8"), source_names, target_names)
        if not remapped:
            continue
        target_image = image_output / image.name
        if target_image.exists():
            target_image = image_output / f"{source.name}_{image.name}"
        shutil.copy2(image, target_image)
        target_label = label_output / f"{target_image.stem}.txt"
        target_label.write_text(remapped, encoding="utf-8")
        for line in remapped.splitlines():
            counts[target_names[int(line.split()[0])]] += 1
    (output / "classes.json").write_text(json.dumps(target_names, ensure_ascii=False, indent=2), encoding="utf-8")
    return {name: count for name, count in counts.items() if count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Siapkan sumber YOLO menjadi tiga kelas global FruitVision.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True, help="JSON list nama kelas sumber.")
    parser.add_argument("--target", nargs=3, required=True)
    args = parser.parse_args()
    source_names = load_source_names(args.names)
    print(json.dumps(prepare_yolo_source(args.source, args.output, source_names, args.target), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
