from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

PRESET_EPOCHS = (50, 60, 70, 80, 90, 100)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "combined-top3-cc0-targeted" / "data.yaml"
DEFAULT_BASE_WEIGHTS = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "loop-detector-yolo26n"
    / "experiment-512-yolo26n"
    / "weights"
    / "best.pt"
)
PROJECT = ROOT / "runs" / "detect" / "runs" / "loop-detector-yolo26n-targeted"


def validate_epoch_count(value: int) -> int:
    if value <= 0:
        raise ValueError("jumlah epoch harus berupa bilangan positif")
    return value


def build_run_name(epochs: int) -> str:
    validate_epoch_count(epochs)
    return f"continuation-512-yolo26n-from-epoch30-{epochs}ep"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_training(epochs: int, base_weights: Path | None = None, run_name: str | None = None) -> int:
    import torch
    from ultralytics import YOLO

    from research.yolo26m_training import detector_training_config, validate_detector_dataset

    epochs = validate_epoch_count(epochs)
    weights = (base_weights or Path(os.getenv("FRUITVISION_BASE_WEIGHTS", str(DEFAULT_BASE_WEIGHTS)))).resolve()
    name = run_name or build_run_name(epochs)

    validation = validate_detector_dataset(DATA)
    if not weights.is_file() or weights.stat().st_size == 0:
        raise SystemExit(f"bobot awal tidak ditemukan: {weights}")
    print(f"DATASET {validation}", flush=True)
    print(f"BASE_WEIGHTS {weights} {weights.stat().st_size} {sha256_file(weights)}", flush=True)

    config = detector_training_config()
    torch.set_num_threads(int(config["threads"]))
    torch.set_num_interop_threads(min(8, int(config["threads"])))
    print(
        f"ACCELERATOR device={config['device']} cuda={torch.cuda.is_available()} "
        f"threads={torch.get_num_threads()}",
        flush=True,
    )

    model = YOLO(str(weights))
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=512,
        batch=2,
        workers=0,
        device=config["device"],
        project=str(PROJECT),
        name=name,
        exist_ok=False,
        patience=8,
        plots=False,
        val=True,
        seed=0,
        cache="disk",
        close_mosaic=5,
        lr0=0.00035,
        lrf=0.01,
        warmup_epochs=1.0,
        verbose=True,
    )

    best = PROJECT / name / "weights" / "best.pt"
    if not best.is_file() or best.stat().st_size == 0:
        raise SystemExit(f"best.pt tidak ditemukan: {best}")
    print(f"BEST {best} {best.stat().st_size} {sha256_file(best)}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tuning YOLO26n dari checkpoint epoch 30 dengan jumlah epoch pilihan."
    )
    parser.add_argument("--epochs", type=int, required=True, help="jumlah epoch tambahan, misalnya 50 atau 100")
    parser.add_argument("--base-weights", type=Path, default=None, help="path checkpoint awal, opsional")
    parser.add_argument("--run-name", default=None, help="nama folder hasil, opsional")
    args = parser.parse_args(argv)
    return run_training(args.epochs, args.base_weights, args.run_name)


if __name__ == "__main__":
    raise SystemExit(main())
