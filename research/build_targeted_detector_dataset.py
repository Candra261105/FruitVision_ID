from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


TARGET_CLASS = 2  # green beans in the fixed three-class dataset
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _copy_split(source: Path, output: Path, split: str) -> int:
    image_source = source / "images" / split
    label_source = source / "labels" / split
    image_target = output / "images" / split
    label_target = output / "labels" / split
    image_target.mkdir(parents=True, exist_ok=True)
    label_target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image in _images(image_source):
        label = label_source / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"label tidak ditemukan untuk {image}: {label}")
        shutil.copy2(image, image_target / image.name)
        shutil.copy2(label, label_target / label.name)
        copied += 1
    return copied


def _contains_target(label: Path) -> bool:
    return any(line.split() and line.split()[0] == str(TARGET_CLASS) for line in label.read_text(encoding="utf-8").splitlines())


def _duplicate_target_train(source: Path, output: Path, copies: int) -> int:
    image_source = source / "images" / "train"
    label_source = source / "labels" / "train"
    image_target = output / "images" / "train"
    label_target = output / "labels" / "train"
    candidates = [
        image
        for image in _images(image_source)
        if _contains_target(label_source / f"{image.stem}.txt")
    ]
    for index in range(copies):
        image = candidates[index % len(candidates)]
        label = label_source / f"{image.stem}.txt"
        stem = f"gbboost-{index:05d}-{image.stem}"
        shutil.copy2(image, image_target / f"{stem}{image.suffix}")
        shutil.copy2(label, label_target / f"{stem}.txt")
    return copies


def build_targeted_dataset(
    source: Path,
    output: Path,
    report: Path,
    target_extra_fraction: float = 0.5,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("output harus berbeda dari dataset sumber")
    if target_extra_fraction < 0:
        raise ValueError("target_extra_fraction tidak boleh negatif")

    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "test", "external_holdout"):
        _copy_split(source, output, split)

    source_yaml = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8")) or {}
    source_yaml["path"] = str(output).replace("\\", "/")
    (output / "data.yaml").write_text(yaml.safe_dump(source_yaml, sort_keys=False), encoding="utf-8")
    external_yaml = dict(source_yaml)
    external_yaml["test"] = "images/external_holdout"
    (output / "data-external.yaml").write_text(yaml.safe_dump(external_yaml, sort_keys=False), encoding="utf-8")

    base_train_images = sorted((output / "images" / "train").glob("*"))
    target_train_images = [
        image
        for image in (source / "images" / "train").iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_SUFFIXES
        and _contains_target(source / "labels" / "train" / f"{image.stem}.txt")
    ]
    extra_count = round(len(target_train_images) * target_extra_fraction)
    duplicated = _duplicate_target_train(source, output, extra_count)

    unchanged_splits: dict[str, bool] = {}
    for split in ("val", "test", "external_holdout"):
        source_images = _images(source / "images" / split)
        output_images = _images(output / "images" / split)
        source_labels = sorted((source / "labels" / split).glob("*.txt"))
        output_labels = sorted((output / "labels" / split).glob("*.txt"))
        unchanged_splits[split] = (
            [p.name for p in source_images] == [p.name for p in output_images]
            and [p.name for p in source_labels] == [p.name for p in output_labels]
            and all(_file_hash(a) == _file_hash(b) for a, b in zip(source_images, output_images))
            and all(_file_hash(a) == _file_hash(b) for a, b in zip(source_labels, output_labels))
        )

    result: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "base_train_images": len(base_train_images),
        "green_beans_train_images": len(target_train_images),
        "duplicated_green_beans_images": duplicated,
        "target_extra_fraction": target_extra_fraction,
        "unchanged_splits": unchanged_splits,
        "policy": "train-only green beans oversampling by deterministic exact-image copies; val/test/external_holdout unchanged",
    }
    (output / "EXPERIMENT_SCOPE.md").write_text(
        "# Targeted detector experiment\n\n"
        "Train memakai salinan train baseline dan tambahan salinan deterministik gambar yang memiliki label `green beans` untuk memperbaiki ketidakseimbangan kelas.\n\n"
        "Validation, test utama, dan external holdout disalin tanpa modifikasi dan tidak dicampur ke train.\n\n"
        f"Tambahan train: {duplicated} gambar.\n",
        encoding="utf-8",
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-extra-fraction", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(build_targeted_dataset(args.source, args.output, args.report, args.target_extra_fraction), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
