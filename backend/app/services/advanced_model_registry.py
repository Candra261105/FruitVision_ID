from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image


TASKS = ("disease", "maturity", "rot", "grade")
AdvancedStatus = Literal["eligible", "unavailable", "rejected", "blocked"]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TASK_LABELS = {
    "disease": {"healthy", "unknown", "belly_rot", "pythium_fruit_rot"},
    "maturity": {"unripe", "half_ripe", "ripe"},
    "rot": {"not_rotten", "rotten"},
    "grade": {"A", "B", "C", "reject"},
}
MIN_MACRO_F1 = 0.75
MIN_BALANCED_ACCURACY = 0.75
MIN_SUPPORT = 20
MIN_CLASS_SUPPORT = 5


@dataclass(frozen=True)
class ConditionPrediction:
    label: str
    confidence: float
    margin: float


class AdvancedModelRuntime:
    """Model kondisi per task; bobot dimuat malas saat prediksi pertama."""

    def __init__(
        self,
        task: str,
        status: AdvancedStatus,
        model_path: Path | None = None,
        detail: str = "",
        labels: tuple[str, ...] = (),
        fruits: tuple[str, ...] = (),
        imgsz: int = 160,
        macro_f1: float | None = None,
    ) -> None:
        self.task = task
        self.status = status
        self.model_path = model_path
        self.detail = detail
        self.labels = labels
        self.fruits = fruits
        self.imgsz = imgsz
        self.macro_f1 = macro_f1
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def runnable(self) -> bool:
        return self.status == "eligible" and self.model_path is not None and bool(self.fruits)

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, crop: Image.Image) -> ConditionPrediction:
        if not self.runnable:
            raise RuntimeError("model task belum eligible atau tidak punya fruit target")
        with self._lock:
            model = self._load()
            results = model.predict(source=crop, imgsz=self.imgsz, verbose=False, device="cpu")
        if not results or results[0].probs is None:
            raise RuntimeError("model tidak mengembalikan probabilitas")
        result = results[0]
        probabilities = result.probs
        values = [float(value) for value in probabilities.data.tolist()]
        top1 = int(probabilities.top1)
        names = result.names
        label = str(names[top1]) if isinstance(names, dict) else str(names[top1])
        top = values[top1] if top1 < len(values) else float(probabilities.top1conf)
        second = max((value for index, value in enumerate(values) if index != top1), default=0.0)
        return ConditionPrediction(label=label, confidence=top, margin=top - second)


def _rejected(task: str, model_path: Path | None, detail: str) -> AdvancedModelRuntime:
    return AdvancedModelRuntime(task, "rejected", model_path, detail)


def _build_runtime(task: str, model_path: Path, gate: dict[str, Any]) -> AdvancedModelRuntime:
    provenance = gate.get("provenance", {})
    evaluation = gate.get("evaluation", {})
    if not isinstance(provenance, dict) or not isinstance(evaluation, dict):
        return _rejected(task, model_path, "Schema gate/provenance tidak valid.")
    metrics_valid = all(
        isinstance(evaluation.get(field), (int, float))
        and not isinstance(evaluation.get(field), bool)
        and math.isfinite(evaluation[field])
        and 0 <= evaluation[field] <= 1
        for field in ("macro_f1", "balanced_accuracy")
    )
    per_class = evaluation.get("per_class", {})
    evaluated_labels = evaluation.get("labels", [])
    labels_valid = (
        isinstance(evaluated_labels, list)
        and len(evaluated_labels) >= 2
        and len(set(evaluated_labels)) == len(evaluated_labels)
        and set(evaluated_labels) <= TASK_LABELS[task]
    )
    support_valid = (
        isinstance(evaluation.get("support"), int)
        and evaluation["support"] >= MIN_SUPPORT
        and isinstance(per_class, dict)
        and labels_valid
        and all(
            isinstance(per_class.get(label), dict)
            and isinstance(per_class[label].get("support"), int)
            and per_class[label]["support"] >= MIN_CLASS_SUPPORT
            for label in evaluated_labels
        )
    )
    if gate.get("status") != "eligible":
        return _rejected(task, model_path, "Model belum lolos gate.")
    taxonomy_ok = provenance.get("taxonomy_version") == 1
    if not isinstance(provenance.get("taxonomy_version"), int):
        taxonomy_ok = False
    taxonomy_sha = provenance.get("taxonomy_sha256")
    if (
        provenance.get("task") != task
        or not taxonomy_ok
        or not isinstance(taxonomy_sha, str)
        or not SHA256.fullmatch(taxonomy_sha)
        or not isinstance(provenance.get("split_strategy"), str)
        or not provenance["split_strategy"].strip()
        or any(not isinstance(provenance.get(field), str) or not SHA256.fullmatch(provenance[field]) for field in ("manifest_sha256", "model_sha256"))
        or provenance.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest()
        or evaluation.get("task") != task
        or not labels_valid
        or not metrics_valid
        or evaluation.get("macro_f1", 0) < MIN_MACRO_F1
        or evaluation.get("balanced_accuracy", 0) < MIN_BALANCED_ACCURACY
        or not support_valid
    ):
        return _rejected(task, model_path, "Provenance atau evaluasi model tidak cocok.")

    fruits_raw = provenance.get("fruits", [])
    fruits = tuple(str(item).strip().lower() for item in fruits_raw) if isinstance(fruits_raw, list) else ()
    fruits = tuple(fruit for fruit in fruits if fruit in {"cucumber", "tomato", "green_beans"})
    imgsz_raw = provenance.get("imgsz", 160)
    imgsz = imgsz_raw if isinstance(imgsz_raw, int) and 32 <= imgsz_raw <= 640 else 160
    macro_f1 = float(evaluation["macro_f1"])
    detail = (
        f"Model {task} aktif untuk {', '.join(fruits) if fruits else 'tanpa fruit target'} "
        f"(macro-F1 {macro_f1:.2f} pada test holdout)."
    )
    return AdvancedModelRuntime(task, "eligible", model_path, detail, tuple(evaluated_labels), fruits, imgsz, macro_f1)


def load_advanced_models(model_root: Path) -> dict[str, AdvancedModelRuntime]:
    result: dict[str, AdvancedModelRuntime] = {}
    taxonomy_candidates = (
        model_root.parent / "research" / "advanced_taxonomy.yaml",
        model_root / "research" / "advanced_taxonomy.yaml",
    )
    taxonomy_path = next((candidate for candidate in taxonomy_candidates if candidate.is_file()), taxonomy_candidates[0])
    taxonomy_sha256 = hashlib.sha256(taxonomy_path.read_bytes()).hexdigest() if taxonomy_path.is_file() else None
    for task in TASKS:
        model_path = model_root / f"{task}.pt"
        gate_path = model_root / f"{task}.gate.json"
        if not model_path.is_file() or not gate_path.is_file():
            result[task] = AdvancedModelRuntime(task, "unavailable", detail="Model task belum tersedia.")
            continue
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result[task] = _rejected(task, model_path, "Gate model tidak valid.")
            continue
        if not isinstance(gate, dict):
            result[task] = _rejected(task, model_path, "Gate model harus berupa object JSON.")
            continue
        provenance = gate.get("provenance", {})
        if (
            not isinstance(provenance, dict)
            or taxonomy_sha256 is None
            or provenance.get("taxonomy_sha256") != taxonomy_sha256
        ):
            result[task] = _rejected(task, model_path, "Taxonomy berubah sejak model dievaluasi; jalankan ulang evaluasi.")
            continue
        result[task] = _build_runtime(task, model_path, gate)
    return result
