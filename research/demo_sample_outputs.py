from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from app.config import load_settings
from app.services.advanced_model_registry import load_advanced_models
from app.services.model_registry import load_runtime
from app.services.prediction_service import PredictionService

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research" / "reports"

CASES = [
    ("tomato", ROOT / "data/processed/kitchen-top3/images/IMG_004.jpg", None),
    ("cucumber", ROOT / "data/processed/kitchen-top3/images/IMG_003.jpg", None),
    ("green_beans", ROOT / "data/processed/bangladesh/images/bangladesh_000003.jpg", None),
    ("no_fruit", ROOT / "data/processed/kitchen-top3/images/IMG_001.jpg", None),
    ("dark_tomato", ROOT / "data/processed/kitchen-top3/images/IMG_004.jpg", "dark"),
    ("blurry_cucumber", ROOT / "data/processed/kitchen-top3/images/IMG_003.jpg", "blur"),
]


def variant(image: Image.Image, kind: str | None) -> Image.Image:
    if kind == "dark":
        return ImageEnhance.Brightness(image).enhance(0.22)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.4))
    return image


def main() -> int:
    settings = load_settings()
    runtime = load_runtime(settings.model_path, settings.confidence_threshold)
    advanced = load_advanced_models(settings.model_path.parent)
    service = PredictionService(
        provider=runtime.provider,
        model_loaded=runtime.model_loaded,
        mode=runtime.mode,
        threshold=settings.confidence_threshold,
        advanced_models=advanced,
    )
    print("mode=", runtime.mode, "model_loaded=", runtime.model_loaded)
    print("advanced=", {task: models.status for task, models in advanced.items()})

    rows = []
    for name, path, kind in CASES:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        image = variant(image, kind)
        result = service.predict(image)
        detections = [
            {
                "fruit": det.fruit,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "stage": det.stage,
                "stage_available": det.stage_available,
                "review_status": det.review_status,
                "warnings": det.warnings,
                "conditions": {
                    task: {
                        "status": cond.status,
                        "label": cond.label,
                        "confidence": cond.confidence,
                        "detail": cond.detail,
                    }
                    for task, cond in det.conditions.items()
                },
            }
            for det in result.detections
        ]
        row = {
            "case": name,
            "input": str(path.relative_to(ROOT)).replace("\\", "/"),
            "variant": kind or "asli",
            "message": result.message,
            "enhancement_status": result.enhancement_status,
            "enhancement_steps": result.enhancement_steps,
            "quality": result.quality.model_dump(),
            "detection_count": len(detections),
            "detections": detections,
            "capabilities": {task: capability.status for task, capability in result.capabilities.items()},
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("case", "message", "detection_count", "detections")}, ensure_ascii=False, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": runtime.mode,
        "confidence_threshold": settings.confidence_threshold,
        "cases": rows,
    }
    (OUT_DIR / "sample_outputs_20261118.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Contoh output pipeline per jenis input",
        "",
        f"Mode detector: `{runtime.mode}` · threshold keyakinan {settings.confidence_threshold}",
        "",
        "| Kasus | Input | Objek terdeteksi | Pesan | Enhancement |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        fruits = ", ".join(f"{d['fruit']} {round(d['confidence']*100)}%" for d in row["detections"]) or "-"
        lines.append(
            f"| {row['case']} | `{Path(row['input']).name}` | {fruits} | {row['message']} | {row['enhancement_status']} |"
        )
    lines.extend([
        "",
        "## Status fitur lanjutan saat pengujian",
        "",
        "| Fitur | Status |",
        "|---|---|",
    ])
    for task, capability in rows[0]["capabilities"].items():
        lines.append(f"| {task} | {capability} |")
    (OUT_DIR / "sample_outputs_20261118.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
