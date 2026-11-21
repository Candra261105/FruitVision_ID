from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_evaluation_result(metrics, names, model, data, device, elapsed):
    box = metrics.box
    names = names if isinstance(names, dict) else {index: value for index, value in enumerate(names)}
    per_class = {}
    for position, class_index in enumerate(getattr(box, "ap_class_index", [])):
        per_class[str(names.get(int(class_index), class_index))] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "map50": float(box.ap50[position]),
            "map50_95": float(box.ap[position]),
        }
    return {
        "model": str(model),
        "data": str(data),
        "device": device,
        "elapsed_seconds": round(elapsed, 3),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "limitation": "Metrik publik tidak membuktikan performa pada kamera laptop pengguna.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi bobot YOLO dan simpan metrik.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("research/reports/evaluation.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics belum terpasang. Pasang dengan: py -m pip install ultralytics") from exc
    model = YOLO(str(args.model))
    started = time.perf_counter()
    metrics = model.val(data=str(args.data), device=args.device, verbose=False)
    elapsed = time.perf_counter() - started
    result = build_evaluation_result(metrics, getattr(model, "names", {}), args.model, args.data, args.device, elapsed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
