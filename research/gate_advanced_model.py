from __future__ import annotations

import json
import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIN_MACRO_F1 = 0.75
MIN_BALANCED_ACCURACY = 0.75
MIN_SUPPORT = 20
MIN_CLASS_SUPPORT = 5
TASK_LABELS = {
    "disease": {"healthy", "unknown", "belly_rot", "pythium_fruit_rot"},
    "maturity": {"unripe", "half_ripe", "ripe"},
    "rot": {"not_rotten", "rotten"},
    "grade": {"A", "B", "C", "reject"},
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def write_advanced_gate(model_path: Path, evaluation: dict[str, Any], provenance: dict[str, Any]) -> Path:
    reasons: list[str] = []
    if not model_path.is_file():
        reasons.append("model file tidak ditemukan")
    task = provenance.get("task")
    if task not in TASK_LABELS:
        reasons.append("task provenance tidak diizinkan")
    if evaluation.get("task") != task:
        reasons.append("evaluation.task tidak cocok dengan provenance.task")
    for field in ("manifest_sha256", "taxonomy_sha256", "model_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            reasons.append(f"{field} harus SHA-256 heksadesimal")
    if model_path.is_file() and isinstance(provenance.get("model_sha256"), str) and SHA256.fullmatch(provenance["model_sha256"]):
        if _sha256(model_path) != provenance["model_sha256"]:
            reasons.append("model_sha256 tidak cocok dengan file model")
    if not isinstance(provenance.get("taxonomy_version"), int) or provenance["taxonomy_version"] != 1:
        reasons.append("taxonomy_version tidak cocok")
    if not isinstance(provenance.get("split_strategy"), str) or not provenance["split_strategy"].strip():
        reasons.append("split_strategy tidak ada")
    labels = evaluation.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) < 2
        or len(set(labels)) != len(labels)
        or not set(labels) <= TASK_LABELS.get(task, set())
    ):
        reasons.append("label evaluasi tidak cocok dengan taxonomy")
    for metric in ("macro_f1", "balanced_accuracy"):
        value = evaluation.get(metric)
        if not _valid_metric(value):
            reasons.append(f"{metric} tidak finite atau di luar rentang")
    macro_f1 = evaluation.get("macro_f1", 0.0)
    balanced_accuracy = evaluation.get("balanced_accuracy", 0.0)
    if not _valid_metric(macro_f1) or macro_f1 < MIN_MACRO_F1:
        reasons.append("macro_f1 di bawah minimum")
    if not _valid_metric(balanced_accuracy) or balanced_accuracy < MIN_BALANCED_ACCURACY:
        reasons.append("balanced_accuracy di bawah minimum")
    if not isinstance(evaluation.get("support"), int) or int(evaluation.get("support", 0)) < MIN_SUPPORT:
        reasons.append("support test terlalu kecil")
    per_class = evaluation.get("per_class", {})
    evaluated_labels = [str(label) for label in labels] if isinstance(labels, list) else []
    if not isinstance(per_class, dict) or any(
        not isinstance(per_class.get(label), dict)
        or not isinstance(per_class[label].get("support"), int)
        or per_class[label]["support"] < MIN_CLASS_SUPPORT
        for label in evaluated_labels
    ):
        reasons.append("support per kelas terlalu kecil atau tidak lengkap")
    payload = {
        "status": "eligible" if not reasons else "rejected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_path.name,
        "evaluation": evaluation,
        "provenance": provenance,
        "reasons": reasons,
    }
    gate_path = model_path.with_suffix(".gate.json")
    gate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate_path
