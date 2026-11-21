from __future__ import annotations

import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"C:\Users\dragon\Downloads\FruitVision_ID")
DATA = ROOT / "data" / "processed" / "combined-top3-final" / "data.yaml"

PROMPT_SETS = {
    "plain": ["cucumber", "tomato", "green beans"],
    "photo_of": ["a photo of a cucumber", "a photo of a tomato", "a photo of green beans"],
    "fruit_words": ["cucumber fruit", "tomato fruit", "green bean pod"],
}


def run(name: str, prompts: list[str]) -> dict:
    model = YOLO(str(ROOT / "models" / "yolov8s-world.pt"))
    model.set_classes(prompts)
    result = model.val(
        data=str(DATA),
        split="test",
        imgsz=512,
        device="cpu",
        workers=0,
        plots=False,
        verbose=False,
        conf=0.05,
    )
    box = result.box
    ap50 = getattr(box, "ap50", None)
    index = list(getattr(box, "ap_class_index", []))
    per_class = {}
    names = model.names
    if ap50 is not None and len(ap50) == len(index):
        for position, cls in enumerate(index):
            per_class[str(names[int(cls)])] = round(float(ap50[position]), 4)
    return {
        "prompts": prompts,
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
    }


def main() -> int:
    out = {}
    for name, prompts in PROMPT_SETS.items():
        print(f"== {name}", flush=True)
        out[name] = run(name, prompts)
        print(json.dumps(out[name], ensure_ascii=False), flush=True)
    target = ROOT / "research" / "reports" / "yoloworld_prompt_sweep_20261118.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
