from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"C:\Users\dragon\Downloads\FruitVision_ID")
DATA = ROOT / "data" / "processed" / "combined-top3-final" / "data.yaml"


def summarize(model: YOLO, result) -> dict:
    box = result.box
    names = model.names
    index = list(getattr(box, "ap_class_index", []))
    ap50 = getattr(box, "ap50", None)
    per_class = {}
    if ap50 is not None and len(ap50) == len(index):
        for position, cls in enumerate(index):
            per_class[str(names[int(cls)])] = round(float(ap50[position]), 4)
    return {
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
    }


def run(tag: str, weights: Path, imgsz: int, prompts: list[str] | None) -> dict:
    model = YOLO(str(weights))
    if prompts:
        model.set_classes(prompts)
    result = model.val(
        data=str(DATA),
        split="test",
        imgsz=imgsz,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False,
    )
    payload = {"tag": tag, "weights": weights.name, "imgsz": imgsz, **summarize(model, result)}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def main() -> int:
    out = {}
    out["world_1024"] = run(
        "world_1024",
        ROOT / "models" / "yolov8s-world.pt",
        1024,
        ["cucumber", "tomato", "green beans"],
    )
    legacy = ROOT / "runs" / "fruitvision-loop" / "yolo11n-light-original" / "weights" / "best.pt"
    out["detector_640"] = run("detector_640", legacy, 640, None)
    out["detector_1024"] = run("detector_1024", legacy, 1024, None)
    target = ROOT / "research" / "reports" / "resolution_sweep_20261118.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
