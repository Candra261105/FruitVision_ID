from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from research.build_combined_dataset import remap_label_lines


def _copy_split(
    source: Path,
    output: Path,
    split: str,
    destination_split: str | None = None,
    prefix: str = "",
    source_names: list[str] | None = None,
    target_names: list[str] | None = None,
) -> int:
    destination_split = destination_split or split
    image_source = source / "images" / split
    label_source = source / "labels" / split
    image_destination = output / "images" / destination_split
    label_destination = output / "labels" / destination_split
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for image in sorted(image_source.glob("*")) if image_source.is_dir() else []:
        if not image.is_file():
            continue
        label = label_source / f"{image.stem}.txt"
        if not label.is_file():
            continue
        label_text = label.read_text(encoding="utf-8")
        if source_names is not None and target_names is not None:
            label_text = remap_label_lines(label_text, source_names, target_names)
            if not label_text:
                continue
        target_stem = f"{prefix}{image.stem}"
        shutil.copy2(image, image_destination / f"{target_stem}{image.suffix}")
        (label_destination / f"{target_stem}.txt").write_text(label_text, encoding="utf-8")
        count += 1
    return count


def build_detector_experiment(base: Path, extra: Path, output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    for generated_split in ("train", "val", "test", "external_holdout"):
        for folder in (output / "images" / generated_split, output / "labels" / generated_split):
            if folder.is_dir():
                shutil.rmtree(folder)
    data_yaml = base / "data.yaml"
    target_names: list[str] = []
    if data_yaml.is_file():
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        target_names = [str(name) for name in payload.get("names", [])]
        payload["path"] = str(output.resolve()).replace("\\", "/")
        (output / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    source_names: list[str] | None = None
    source_classes = extra / "classes.json"
    if source_classes.is_file():
        raw_names = __import__("json").loads(source_classes.read_text(encoding="utf-8"))
        source_names = [str(value) for value in raw_names.values()] if isinstance(raw_names, dict) else [str(value) for value in raw_names]
    result = {
        "base_train": _copy_split(base, output, "train"),
        "extra_train": _copy_split(extra, output, "train", prefix="extra-", source_names=source_names, target_names=target_names),
        "base_val": _copy_split(base, output, "val"),
        "base_test": _copy_split(base, output, "test"),
        "extra_holdout": _copy_split(extra, output, "holdout", "external_holdout", prefix="extra-", source_names=source_names, target_names=target_names),
    }
    manifest = output / "EXPERIMENT_SCOPE.md"
    manifest.write_text(
        "# Detector experiment scope\n\n"
        "Train: base train + extra train.\n"
        "Validation/test: base val/test tetap, tanpa data extra.\n"
        "External holdout: extra holdout disimpan untuk evaluasi terpisah.\n",
        encoding="utf-8",
    )
    return result
