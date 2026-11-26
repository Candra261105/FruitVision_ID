"""Evaluate a trained checkpoint on real-only test and external holdout data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

CLASS_NAMES = ("tomato", "pumpkin", "lemon")
MIN_SUPPORT = 1
MIN_CLASS_MAP50 = 0.40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(data_yaml: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_yaml.parent / path


def validate_evaluation_inputs(data_yaml: Path, manifest: Path, external_yaml: Path | None = None) -> None:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data YAML tidak ditemukan: {data_yaml}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest tidak ditemukan: {manifest}")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names, key=lambda item: int(item))]
    if tuple(names or ()) != CLASS_NAMES:
        raise ValueError(f"class order harus tepat {CLASS_NAMES}, ditemukan {names}")
    for split in ("train", "val", "test"):
        value = payload.get(split)
        if not isinstance(value, str) or not _resolve_path(data_yaml, value).exists():
            raise FileNotFoundError(f"split {split} tidak ditemukan pada {data_yaml}")
    if external_yaml is not None:
        validate_evaluation_inputs(external_yaml, manifest)


def summarize_metrics(metrics: Any, names: dict[int, str] | list[str] | tuple[str, ...]) -> dict[str, Any]:
    box = metrics.box
    name_map = names if isinstance(names, dict) else dict(enumerate(names))
    per_class: dict[str, dict[str, float]] = {}
    class_indices = getattr(box, "ap_class_index", range(len(getattr(box, "p", []))))
    support_values = getattr(box, "nt_per_class", getattr(box, "support", []))
    for position, class_index in enumerate(class_indices):
        label = str(name_map[int(class_index)])
        support = int(support_values[int(class_index)]) if len(support_values) > int(class_index) else 0
        per_class[label] = {
            "precision": float(box.p[position]),
            "recall": float(box.r[position]),
            "map50": float(box.ap50[position]),
            "map50_95": float(box.ap[position]),
            "support": support,
        }
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
    }


def build_gate(weights: Path, dataset_manifest: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = metrics.get("per_class") or {}
    reasons: list[str] = []
    missing = [name for name in CLASS_NAMES if name not in per_class]
    if missing:
        reasons.append(f"kelas tanpa metrik: {', '.join(missing)}")
    for name in CLASS_NAMES:
        values = per_class.get(name)
        if not values:
            continue
        if values.get("support", 0) < MIN_SUPPORT:
            reasons.append(f"support {name} kurang dari {MIN_SUPPORT}")
        if values.get("map50", 0.0) < MIN_CLASS_MAP50:
            reasons.append(f"mAP50 {name} < {MIN_CLASS_MAP50}")
    status = "partial" if missing else ("rejected" if reasons else "eligible")
    return {
        "schema_version": 1,
        "status": status,
        "classes": list(CLASS_NAMES),
        "weights": str(weights),
        "weights_sha256": _sha256(weights) if weights.is_file() else None,
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": _sha256(dataset_manifest) if dataset_manifest.is_file() else None,
        "metrics": metrics,
        "reasons": reasons,
        "detail": "Model baru tidak otomatis dipasang ke aplikasi. Status eligible tetap memerlukan aktivasi manual.",
    }


def evaluate_model(weights: Path, dataset_yaml: Path, external_yaml: Path | None = None, device: str = "cpu") -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    started = time.perf_counter()
    metrics = model.val(data=str(dataset_yaml), split="test", device=device, workers=0, plots=True, verbose=False)
    result = summarize_metrics(metrics, getattr(model, "names", CLASS_NAMES))
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["dataset"] = str(dataset_yaml)
    if external_yaml is not None:
        external_metrics = model.val(data=str(external_yaml), split="test", device=device, workers=0, plots=True, verbose=False)
        result["external"] = summarize_metrics(external_metrics, getattr(model, "names", CLASS_NAMES))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi test real-only dan gate YOLO26n.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--external-data", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    validate_evaluation_inputs(args.data.resolve(), args.manifest.resolve(), args.external_data.resolve() if args.external_data else None)
    metrics = evaluate_model(args.weights, args.data, args.external_data, args.device)
    gate = build_gate(args.weights, args.manifest, metrics)
    payload = {"metrics": metrics, "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "eligible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
