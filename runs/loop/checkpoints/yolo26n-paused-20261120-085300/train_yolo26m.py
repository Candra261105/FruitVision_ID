from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO

from research.yolo26m_training import detector_training_config, validate_detector_dataset


DATA = ROOT / "data" / "processed" / "combined-top3-cc0-experiment" / "data.yaml"
CHECKPOINT = ROOT / "yolo26n.pt"
PROJECT = ROOT / "runs" / "detect" / "runs" / "loop-detector-yolo26n"
NAME = "experiment-512-yolo26n"


def main() -> int:
    validation = validate_detector_dataset(DATA)
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size == 0:
        raise SystemExit(f"checkpoint yolo26m.pt tidak ditemukan: {CHECKPOINT}")
    print(f"DATASET {validation}", flush=True)
    print(f"CHECKPOINT {CHECKPOINT} {CHECKPOINT.stat().st_size}", flush=True)

    config = detector_training_config()
    torch.set_num_threads(int(config["threads"]))
    torch.set_num_interop_threads(min(8, int(config["threads"])))
    print(f"ACCELERATOR device={config['device']} cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}", flush=True)

    model = YOLO(str(CHECKPOINT))
    model.train(
        data=str(DATA),
        epochs=30,
        imgsz=512,
        batch=2,
        workers=0,
        device=config["device"],
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        patience=10,
        plots=False,
        val=True,
        seed=0,
        cache="disk",
        verbose=True,
    )
    best = PROJECT / NAME / "weights" / "best.pt"
    if not best.is_file() or best.stat().st_size == 0:
        raise SystemExit(f"best.pt tidak ditemukan: {best}")
    print(f"BEST {best} {best.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
