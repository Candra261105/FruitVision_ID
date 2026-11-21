from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.inference.base import InferenceProvider
from app.services.prediction_service import FRUIT_NAMES, split_label

TARGET_CLASS_PROMPTS = ("cucumber", "tomato", "green beans")
TARGET_DATASET_NAMES = ("cucumber", "tomato", "green beans")


@dataclass(frozen=True)
class ModelRuntime:
    provider: InferenceProvider | None
    model_loaded: bool
    mode: str
    name: str
    stage_classes: tuple[str, ...]
    limitation: str


def load_runtime(
    model_path: Path,
    confidence_threshold: float,
    allow_unapproved_custom: bool = False,
) -> ModelRuntime:
    if not model_path.is_file():
        return ModelRuntime(
            provider=None,
            model_loaded=False,
            mode="unavailable",
            name="FruitVision ID belum memiliki bobot model",
            stage_classes=(),
            limitation="Model terlatih belum dipasang. Aplikasi tidak membuat prediksi palsu.",
        )

    is_world_model = "world" in model_path.stem.lower()
    gate_path = model_path.with_suffix(".gate.json")
    gate: dict[str, Any] = {}
    if gate_path.is_file():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gate = {}
    is_trial_model = not is_world_model and allow_unapproved_custom and gate.get("status") != "eligible"
    if not is_world_model and gate.get("status") != "eligible" and not allow_unapproved_custom:
        return ModelRuntime(
            provider=None,
            model_loaded=False,
            mode="unavailable",
            name=f"Model belum lolos gate: {model_path.name}",
            stage_classes=(),
            limitation="Model custom hanya boleh dipakai setelah status evaluasi gate menjadi eligible.",
        )

    try:
        from app.inference.ultralytics_provider import UltralyticsProvider

        provider_kwargs = {"class_prompts": TARGET_CLASS_PROMPTS} if is_world_model else {}
        metrics = gate.get("metrics")
        gate_imgsz = gate.get("imgsz")
        if gate_imgsz is None and isinstance(metrics, dict):
            gate_imgsz = metrics.get("imgsz")
        if isinstance(gate_imgsz, int) and 320 <= gate_imgsz <= 1280:
            provider_kwargs["imgsz"] = gate_imgsz
        provider = UltralyticsProvider(model_path, confidence_threshold, **provider_kwargs)
        available_classes = {split_label(name, None)[0] for name in provider.class_names}
        missing_classes = sorted(set(FRUIT_NAMES) - available_classes)
        if missing_classes and not provider.is_open_vocabulary:
            return ModelRuntime(
                provider=None,
                model_loaded=False,
                mode="unavailable",
                name=f"Bobot tidak lengkap: {model_path.name}",
                stage_classes=(),
                limitation=(
                    "Bobot model belum mencakup kelas: "
                    + ", ".join(missing_classes)
                    + ". Tidak ada prediksi yang dikeluarkan."
                ),
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return ModelRuntime(
            provider=None,
            model_loaded=False,
            mode="unavailable",
            name="Bobot ada, runtime model belum siap",
            stage_classes=(),
            limitation=f"Model belum dapat dimuat ({type(exc).__name__}). Periksa dependensi dan bobot.",
        )

    return ModelRuntime(
        provider=provider,
        model_loaded=True,
        mode="demo" if is_world_model else ("trial" if is_trial_model else "custom"),
        name=model_path.name,
        stage_classes=(),
        limitation=(
            "Bobot YOLO-World adalah fallback demo, bukan model custom tervalidasi."
            if is_world_model
            else (
                "Model YOLO26n sedang dipakai untuk uji coba. Hasilnya belum lolos gate evaluasi."
                if is_trial_model
                else "Model custom lolos gate. Kematangan, penyakit, busuk, dan grade tetap belum tersedia."
            )
        ),
    )


def model_info(runtime: ModelRuntime) -> dict[str, Any]:
    return {
        "name": runtime.name,
        "mode": runtime.mode,
        "model_available": runtime.model_loaded,
        "classes": list(FRUIT_NAMES),
        "stage_classes": list(runtime.stage_classes),
        "limitation": runtime.limitation,
    }
