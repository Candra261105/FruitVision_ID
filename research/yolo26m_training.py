from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import torch


TARGET_CLASSES = ["cucumber", "tomato", "green beans"]


def detector_training_config() -> dict[str, object]:
    """Return a truthful accelerator choice and the requested CPU thread count."""
    return {
        "device": 0 if torch.cuda.is_available() else "cpu",
        "threads": 8,
    }


def _resolve_dataset_path(data_yaml: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_yaml.parent / path


def validate_detector_dataset(data_yaml: Path) -> dict[str, Any]:
    """Validate the exact three-class dataset used by the YOLO26m run."""
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML tidak ditemukan: {data_yaml}")

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names, key=lambda item: int(item))]
    if names != TARGET_CLASSES:
        raise ValueError(
            f"nama kelas harus tepat {TARGET_CLASSES}, ditemukan: {names}"
        )

    dataset_root = Path(payload.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()

    split_paths: dict[str, str] = {}
    for split in ("train", "val", "test"):
        value = payload.get(split)
        if not isinstance(value, str):
            raise ValueError(f"split {split!r} tidak ada di dataset YAML")
        split_path = _resolve_dataset_path(dataset_root / "data.yaml", value)
        if not split_path.exists():
            raise FileNotFoundError(f"split {split!r} tidak ditemukan: {split_path}")
        split_paths[split] = str(split_path)

    return {
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "names": names,
        "splits": split_paths,
    }


def resolve_yolo26m_checkpoint(project: Path, name: str) -> Path:
    """Resolve only the named run's best checkpoint."""
    candidates = (
        project / name / "weights" / "best.pt",
        project / "runs" / "detect" / project.name / name / "weights" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(f"best.pt untuk run {name!r} tidak ditemukan di {project}")
