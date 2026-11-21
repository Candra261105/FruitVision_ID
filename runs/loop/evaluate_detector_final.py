from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

TARGET_CLASSES = {"cucumber", "tomato", "green beans"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(weights: Path, data: Path, split: str, imgsz: int) -> dict[str, Any]:
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split=split, imgsz=imgsz, device="cpu", workers=0, plots=False, verbose=False)
    box = metrics.box
    index = list(getattr(box, "ap_class_index", []))
    ap50 = list(getattr(box, "ap50", []))
    per_class = {str(model.names[int(cls)]): float(ap50[position]) for position, cls in enumerate(index)}
    missing = sorted(TARGET_CLASSES - set(per_class))
    payload: dict[str, Any] = {
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "data": str(data),
        "split": split,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "missing_classes": missing,
        "gate": {
            "status": "eligible" if float(box.map50) >= 0.55 and not missing and all(value >= 0.40 for value in per_class.values()) else "rejected",
            "requirements": {"map50": 0.55, "per_class_ap50": 0.40, "classes": sorted(TARGET_CLASSES)},
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi detector tiga kelas dan tulis provenance.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.weights, args.data, args.split, args.imgsz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "eligible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
