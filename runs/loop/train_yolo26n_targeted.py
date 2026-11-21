from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO

from research.yolo26m_training import detector_training_config, validate_detector_dataset


DATA = ROOT / "data" / "processed" / "combined-top3-cc0-targeted" / "data.yaml"
BASE_WEIGHTS = Path(
    os.getenv(
        "FRUITVISION_BASE_WEIGHTS",
        str(ROOT / "runs" / "detect" / "runs" / "loop-detector-yolo26n" / "experiment-512-yolo26n" / "weights" / "best.pt"),
    )
)
PROJECT = ROOT / "runs" / "detect" / "runs" / "loop-detector-yolo26n-targeted"
NAME = "continuation-512-yolo26n-from-epoch30"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    validation = validate_detector_dataset(DATA)
    if not BASE_WEIGHTS.is_file() or BASE_WEIGHTS.stat().st_size == 0:
        raise SystemExit(f"bobot awal tidak ditemukan: {BASE_WEIGHTS}")
    print(f"DATASET {validation}", flush=True)
    print(f"BASE_WEIGHTS {BASE_WEIGHTS} {BASE_WEIGHTS.stat().st_size} {sha256_file(BASE_WEIGHTS)}", flush=True)

    config = detector_training_config()
    torch.set_num_threads(int(config["threads"]))
    torch.set_num_interop_threads(min(8, int(config["threads"])))
    print(f"ACCELERATOR device={config['device']} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}", flush=True)

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATA),
        epochs=20,
        imgsz=512,
        batch=2,
        workers=0,
        device=config["device"],
        project=str(PROJECT),
        name=NAME,
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
    best = PROJECT / NAME / "weights" / "best.pt"
    if not best.is_file() or best.stat().st_size == 0:
        raise SystemExit(f"best.pt tidak ditemukan: {best}")
    print(f"BEST {best} {best.stat().st_size} {sha256_file(best)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
