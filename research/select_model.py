from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_model(
    runs_root: Path,
    required_classes: list[str],
    min_class_recall: float = 0.3,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_class = metrics.get("per_class", {})
        reasons: list[str] = []
        missing = [name for name in required_classes if name not in per_class]
        if missing:
            reasons.append(f"kelas tidak memiliki metrik: {', '.join(missing)}")
        for name in required_classes:
            recall = per_class.get(name, {}).get("recall")
            if recall is not None and float(recall) < min_class_recall:
                reasons.append(f"recall {name} di bawah gate ({float(recall):.4f} < {min_class_recall:.4f})")
        kitchen_path = metrics_path.parent / "kitchen-metrics.json"
        if not kitchen_path.is_file():
            reasons.append("hasil Kitchen belum tersedia")
            status = "candidate"
        else:
            status = "rejected" if reasons else "eligible"
        candidates.append(
            {
                "run": metrics_path.parent.name,
                "status": status,
                "map50": metrics.get("map50"),
                "map50_95": metrics.get("map50_95"),
                "reasons": reasons,
            }
        )
    candidates.sort(key=lambda item: float(item.get("map50_95") or 0.0), reverse=True)
    return {"status": "ok", "candidates": candidates, "best_eligible": next((item for item in candidates if item["status"] == "eligible"), None)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Seleksi kandidat model dengan gate per kelas dan Kitchen.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs=3, required=True)
    parser.add_argument("--min-class-recall", type=float, default=0.3)
    args = parser.parse_args()
    result = select_model(args.runs, args.classes, args.min_class_recall)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
