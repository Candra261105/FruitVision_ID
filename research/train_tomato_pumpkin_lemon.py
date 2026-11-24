"""YOLO26n trainer for the separate tomato/pumpkin/lemon dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tomato_pumpkin_lemon_dataset import CLASS_NAMES, validate_dataset


def _resolve_path(data_yaml: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_yaml.parent / path


def validate_training_request(data_yaml: Path, epochs: int) -> dict[str, Any]:
    if epochs < 1 or epochs > 10000:
        raise ValueError("epochs harus berada antara 1 dan 10000")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names, key=lambda item: int(item))]
    if tuple(names or ()) != CLASS_NAMES:
        raise ValueError(f"class order harus tepat {CLASS_NAMES}, ditemukan {names}")
    for split in ("train", "val", "test"):
        value = payload.get(split)
        if not isinstance(value, str) or not _resolve_path(data_yaml, value).exists():
            raise FileNotFoundError(f"split {split} tidak ditemukan")
    validation = validate_dataset(data_yaml.parent)
    if validation["invalid_label_count"] or validation["cross_split_duplicate_groups"]:
        raise ValueError("dataset gagal validasi label atau memiliki duplicate lintas split")
    return {"epochs": epochs, "data_yaml": str(data_yaml.resolve()), "names": list(CLASS_NAMES), "validation": validation}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> Path:
    request = validate_training_request(args.data.resolve(), args.epochs)
    from ultralytics import YOLO

    weights = args.weights or os.getenv("FRUITVISION_YOLO26N_WEIGHTS", "yolo26n.pt")
    model = YOLO(str(weights))
    device = args.device
    if device == "auto":
        import torch

        device = 0 if torch.cuda.is_available() else "cpu"
    project = args.project.resolve()
    name = args.name or f"tomato-pumpkin-lemon-{args.epochs}ep"
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(project),
        name=name,
        exist_ok=False,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        close_mosaic=min(10, max(0, args.epochs // 10)),
        plots=True,
        val=True,
    )
    best = project / name / "weights" / "best.pt"
    if not best.is_file() or best.stat().st_size == 0:
        raise FileNotFoundError(f"best.pt tidak ditemukan: {best}")
    gate = {
        "status": "pending_evaluation",
        "classes": list(CLASS_NAMES),
        "weights": str(best),
        "weights_sha256": sha256_file(best),
        "epochs": args.epochs,
        "dataset": request,
        "detail": "Checkpoint selesai, tetapi belum eligible untuk deployment sebelum evaluasi test/external.",
    }
    best.with_suffix(".gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Training YOLO26n tomato/pumpkin/lemon.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/tomato-pumpkin-lemon/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/detect/tomato-pumpkin-lemon")
    parser.add_argument("--name")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    request = validate_training_request(args.data.resolve(), args.epochs)
    if args.validate_only:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
