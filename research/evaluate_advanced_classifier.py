"""Evaluasi model klasifikasi kondisi pada test holdout lalu tulis gate.

Contoh:
    python -m research.evaluate_advanced_classifier `
      --task maturity `
      --weights runs/advanced-condition/maturity/weights/best.pt `
      --test-dir runs/advanced-condition/maturity/dataset/test `
      --manifest data/advanced/tomato_maturity_manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.evaluate_advanced import summarize_classification
from research.gate_advanced_model import write_advanced_gate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "maturity": ["unripe", "half_ripe", "ripe"],
    "rot": ["not_rotten", "rotten"],
    "disease": ["healthy", "belly_rot", "pythium_fruit_rot", "unknown"],
    "grade": ["A", "B", "C", "reject"],
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_images(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def predict_paths(weights: Path, files: list[Path], imgsz: int, batch: int) -> tuple[list[str], list[str], list[float]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    results = model.predict(
        source=[str(path) for path in files],
        imgsz=imgsz,
        batch=batch,
        stream=True,
        verbose=False,
        device="cpu",
    )
    for path, result in zip(files, results):
        probabilities = result.probs
        top1 = int(probabilities.top1)
        predicted = str(names[top1]) if isinstance(names, dict) else str(names[top1])
        y_true.append(path.parent.name)
        y_pred.append(predicted)
        confidences.append(float(probabilities.top1conf))
    return y_true, y_pred, confidences


def markdown_report(task: str, metrics: dict[str, Any], gate_status: str, reasons: list[str], notes: list[str]) -> str:
    lines = [
        f"# Evaluasi model {task}",
        "",
        f"- Waktu: {metrics['evaluated_at']}",
        f"- Model: `{metrics['model']}`",
        f"- Test support: {metrics['support']} crop",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- Rata-rata keyakinan top-1: {metrics['mean_confidence']:.4f}",
        f"- Status gate: **{gate_status}**",
        "",
        "| Kelas | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in metrics["per_class"].items():
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['support']} |"
        )
    lines.extend(["", "## Confusion matrix", "", "| Aktual \\ Prediksi | " + " | ".join(metrics["labels"]) + " |", "|---|" + "---|" * len(metrics["labels"])])
    for actual in metrics["labels"]:
        row = metrics["confusion_matrix"][actual]
        lines.append(f"| {actual} | " + " | ".join(str(row[label]) for label in metrics["labels"]) + " |")
    if reasons:
        lines.extend(["", "## Alasan gate menolak", ""] + [f"- {reason}" for reason in reasons])
    lines.extend(["", "## Catatan", ""] + [f"- {note}" for note in notes])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi classifier kondisi dan tulis gate model.")
    parser.add_argument("--task", choices=sorted(LABELS), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "research" / "advanced_taxonomy.yaml")
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dataset", default="ParsaVictor/tomato-ripeness-detection")
    parser.add_argument("--dataset-license", default="CC BY 4.0")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--split-strategy",
        default="holdout dataset sumber; session_id dikelompokkan; 0 overlap lintas split (audit)",
    )
    args = parser.parse_args()

    weights = args.weights if args.weights.is_absolute() else PROJECT_ROOT / args.weights
    test_dir = args.test_dir if args.test_dir.is_absolute() else PROJECT_ROOT / args.test_dir
    manifest = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    taxonomy = args.taxonomy if args.taxonomy.is_absolute() else PROJECT_ROOT / args.taxonomy
    model_out = args.model_out or PROJECT_ROOT / "models" / f"{args.task}.pt"
    report_path = args.report or PROJECT_ROOT / "runs" / "advanced-condition" / args.task / "evaluation.json"

    files = collect_images(test_dir)
    if not files:
        raise SystemExit(f"tidak ada gambar test di {test_dir}")
    known = LABELS[args.task]
    present = sorted({path.parent.name for path in files})
    labels = [label for label in known if label in present]
    unknown = sorted(set(present) - set(known))
    if unknown:
        raise SystemExit(f"folder test tidak dikenal: {unknown}")
    if len(labels) < 2:
        raise SystemExit(f"label test kurang dari dua kelas: {labels}")

    y_true, y_pred, confidences = predict_paths(weights, files, args.imgsz, args.batch)
    metrics = summarize_classification(args.task, y_true, y_pred, labels)
    metrics["labels"] = labels
    metrics["mean_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    metrics["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    metrics["model"] = model_out.name
    metrics["weights_source"] = str(weights.relative_to(PROJECT_ROOT)) if weights.is_relative_to(PROJECT_ROOT) else str(weights)
    metrics["test_dir"] = str(test_dir.relative_to(PROJECT_ROOT)) if test_dir.is_relative_to(PROJECT_ROOT) else str(test_dir)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, model_out)

    task_records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("task") == args.task:
            task_records.append(record)
    fruits = sorted({str(record.get("fruit", "")) for record in task_records if record.get("fruit")})
    provenance = {
        "task": args.task,
        "taxonomy_version": 1,
        "taxonomy_sha256": sha256_file(taxonomy),
        "manifest_sha256": sha256_file(manifest),
        "model_sha256": sha256_file(model_out),
        "split_strategy": args.split_strategy,
        "dataset": args.dataset,
        "dataset_license": args.dataset_license,
        "fruits": fruits,
        "imgsz": args.imgsz,
        "evaluation_split": "test holdout, tanpa gambar augmentasi",
        "augmentation": "augmentasi Ultralytics hanya pada split train",
        "evaluated_at": metrics["evaluated_at"],
    }
    gate_path = write_advanced_gate(model_out, metrics, provenance)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    notes = [
        f"Dataset: {args.dataset} ({args.dataset_license}).",
        f"Test set: {metrics['support']} gambar, {len(labels)} kelas, tanpa augmentasi.",
        "Metrik hanya berlaku pada domain dataset sumber; foto ponsel di luar domain dapat berbeda.",
        "Augmentasi training tidak dihitung sebagai sampel evaluasi.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"metrics": metrics, "provenance": provenance, "gate": gate}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.with_suffix(".md").write_text(
        markdown_report(args.task, metrics, gate["status"], gate.get("reasons", []), notes),
        encoding="utf-8",
    )
    print(json.dumps({"task": args.task, "status": gate["status"], "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"], "support": metrics["support"], "gate": str(gate_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
