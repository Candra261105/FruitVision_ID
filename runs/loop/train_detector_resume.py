from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "combined-top3-cc0-experiment" / "data.yaml"
INITIAL = ROOT / "runs" / "detect" / "runs" / "loop-detector-cc0-fixed" / "experiment-512-fixed" / "weights" / "last.pt"
PROJECT = ROOT / "runs" / "detect" / "runs" / "loop-detector-cc0-resume"
NAME = "experiment-512-resume"


def main() -> int:
    if not DATA.is_file():
        raise SystemExit(f"dataset tidak ditemukan: {DATA}")
    if not INITIAL.is_file():
        raise SystemExit(f"checkpoint tidak ditemukan: {INITIAL}")
    model = YOLO(str(INITIAL))
    model.train(
        data=str(DATA),
        epochs=20,
        imgsz=512,
        batch=8,
        workers=0,
        device="cpu",
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        patience=6,
        plots=False,
        val=True,
        seed=0,
        cache="ram",
        verbose=True,
    )
    best = PROJECT / NAME / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"best.pt tidak ditemukan: {best}")
    print(f"BEST {best} {best.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
