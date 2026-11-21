from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


def split_hashes(hashes: list[str], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict[str, list[str]]:
    values = sorted(set(hashes))
    random.Random(seed).shuffle(values)
    train_end = int(len(values) * train_ratio)
    val_end = train_end + int(len(values) * val_ratio)
    return {"train": values[:train_end], "val": values[train_end:val_end], "test": values[val_end:]}


def build_data_yaml(output: Path, class_names: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump({"path": str(output.parent.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": class_names}, sort_keys=False),
        encoding="utf-8",
    )


def remap_label_lines(label_text: str, source_names: list[str], target_names: list[str]) -> str:
    target_ids = {name: index for index, name in enumerate(target_names)}
    lines: list[str] = []
    for raw_line in label_text.splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        local_id = int(fields[0])
        if not 0 <= local_id < len(source_names):
            continue
        target_id = target_ids.get(source_names[local_id])
        if target_id is None:
            continue
        lines.append(" ".join([str(target_id), *fields[1:]]))
    return "\n".join(lines) + ("\n" if lines else "")


def choose_unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for record in records:
        image_hash = str(record["hash"])
        current = chosen.get(image_hash)
        if current is None or len(str(record.get("label_text", "")).splitlines()) > len(str(current.get("label_text", "")).splitlines()):
            chosen[image_hash] = record
    return sorted(chosen.values(), key=lambda record: (str(record["hash"]), str(record["source"])))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gabungkan folder YOLO yang telah diaudit dengan split bebas duplikasi.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    if isinstance(class_names, dict):
        class_names = list(class_names.values())
    class_names = [str(name) for name in class_names]
    records: list[dict[str, Any]] = []
    for source in args.source:
        source_classes_path = source / "classes.json"
        if not source_classes_path.is_file():
            raise SystemExit(f"classes.json tidak ditemukan untuk sumber: {source}")
        source_names = json.loads(source_classes_path.read_text(encoding="utf-8"))
        if isinstance(source_names, dict):
            source_names = [source_names[str(index)] for index in range(len(source_names))]
        for image in sorted((source / "images").glob("*")):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            label = source / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                continue
            remapped = remap_label_lines(label.read_text(encoding="utf-8"), source_names, class_names)
            if not remapped:
                continue
            records.append({"image": image, "label": label, "label_text": remapped, "hash": file_hash(image), "source": str(source)})
    records = choose_unique_records(records)
    groups = split_hashes([record["hash"] for record in records], args.seed)
    hash_to_split = {image_hash: split for split, values in groups.items() for image_hash in values}
    for split in groups:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(records):
        split = hash_to_split[record["hash"]]
        stem = f"{index:08d}"
        shutil.copy2(record["image"], args.output / "images" / split / f"{stem}{record['image'].suffix.lower()}")
        (args.output / "labels" / split / f"{stem}.txt").write_text(record["label_text"], encoding="utf-8")
        manifest.append({"id": stem, "split": split, "sha256": record["hash"], "source": record["source"]})
    build_data_yaml(args.output / "data.yaml", class_names)
    (args.output / "manifest.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(manifest), "splits": {key: len(value) for key, value in groups.items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
