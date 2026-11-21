from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_matrix(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("experiment matrix harus berupa list")
    ids = [str(item.get("id", "")) for item in experiments]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("id eksperimen harus unik dan tidak kosong")
    return experiments


def should_stop(stop_file: Path) -> bool:
    return stop_file.exists()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        Path(temp_name).replace(path)
    finally:
        temporary = Path(temp_name)
        if temporary.exists():
            temporary.unlink()


def load_existing_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        return None
    try:
        import torch

        torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return checkpoint


def augmentation_kwargs(experiment: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    group = str(experiment.get("augmentation", ""))
    return dict((policy.get("groups") or {}).get(group, {}))


def summarize_metrics(metrics: Any, names: dict[int, str] | list[str] | None = None) -> dict[str, Any]:
    box = getattr(metrics, "box", metrics)
    names = names or {}
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    class_indices = list(getattr(box, "ap_class_index", range(len(getattr(box, "p", [])))))
    precision = list(getattr(box, "p", []))
    recall = list(getattr(box, "r", []))
    ap50 = list(getattr(box, "ap50", []))
    ap = list(getattr(box, "ap", []))
    per_class: dict[str, Any] = {}
    for position, class_index in enumerate(class_indices):
        class_name = str(names.get(int(class_index), class_index))
        per_class[class_name] = {
            "precision": float(precision[position]) if position < len(precision) else None,
            "recall": float(recall[position]) if position < len(recall) else None,
            "map50": float(ap50[position]) if position < len(ap50) else None,
            "map50_95": float(ap[position]) if position < len(ap) else None,
        }
    return {
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "per_class": per_class,
    }


def train_one(experiment: dict[str, Any], data: Path, output: Path, device: str) -> dict[str, Any]:
    from ultralytics import YOLO

    run_dir = output / str(experiment["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    existing = load_existing_status(status_path)
    if existing.get("status") == "completed":
        return existing
    config = {**experiment, "data": str(data), "device": device}
    write_json_atomic(run_dir / "config.json", config)
    status = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "config": config}
    write_json_atomic(status_path, status)
    resume_checkpoint = find_resume_checkpoint(run_dir)
    model = YOLO(str(resume_checkpoint or experiment["model"]))
    policy_path = Path(str(experiment.get("augmentation_policy", "")))
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) if policy_path.is_file() else {}
    aug = augmentation_kwargs(experiment, policy)
    try:
        if resume_checkpoint:
            result = model.train(resume=True)
        else:
            result = model.train(
                data=str(data),
                epochs=int(experiment["epochs"]),
                imgsz=int(experiment["imgsz"]),
                patience=int(experiment["patience"]),
                device=device,
                seed=int(experiment["seed"]),
                project=str(output.resolve()),
                name=str(experiment["id"]),
                exist_ok=True,
                plots=True,
                **aug,
            )
    except KeyboardInterrupt:
        status.update({"status": "interrupted", "finished_at": datetime.now(timezone.utc).isoformat()})
        write_json_atomic(status_path, status)
        raise
    except Exception as exc:
        status.update({"status": "failed", "finished_at": datetime.now(timezone.utc).isoformat(), "error": type(exc).__name__})
        write_json_atomic(status_path, status)
        raise
    names = getattr(model, "names", {})
    metrics = summarize_metrics(result, names)
    write_json_atomic(run_dir / "metrics.json", metrics)
    status.update({"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics})
    write_json_atomic(status_path, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Loop training YOLO yang dapat dihentikan dan dilanjutkan.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-file", type=Path, default=Path("STOP_TRAINING"))
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"data.yaml tidak ditemukan: {args.data}")
    experiments = load_matrix(args.matrix)
    completed = 0
    for experiment in experiments:
        if args.max_runs and completed >= args.max_runs:
            break
        if should_stop(args.stop_file):
            print(f"Training dihentikan oleh file: {args.stop_file}")
            break
        train_one(experiment, args.data, args.output, args.device)
        completed += 1
    print(f"Eksperimen diproses: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
