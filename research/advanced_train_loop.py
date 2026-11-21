from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from research.advanced_manifest_schema import validate_manifest
from research.advanced_training_status import write_status_atomic


TASKS = {"disease", "maturity", "rot", "grade"}
SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def load_advanced_matrix(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("matrix advanced harus memiliki tasks")
    ids = [str(item.get("id", "")) for item in tasks]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("id task harus unik dan tidak kosong")
    return tasks


def _manifest_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_valid_checkpoint(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        return None
    try:
        import torch

        try:
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception:
            # Checkpoint Ultralytics membawa objek non-tensor; file ini dibuat sendiri oleh training lokal.
            torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return checkpoint


def _safe_name(value: str) -> str:
    return SAFE_NAME.sub("_", value).strip("._") or "item"


class StopFileCallback:
    """Hentikan training di batas epoch tanpa mematikan proses utama."""

    def __init__(self, stop_file: Path) -> None:
        self.stop_file = stop_file
        self.triggered = False

    def __call__(self, trainer: Any) -> None:
        if self.stop_file.exists():
            trainer.stop = True
            self.triggered = True


def _build_task_dataset(records: list[dict[str, Any]], image_root: Path, output: Path) -> tuple[Path, dict[str, int]]:
    """Materialise validated object crops using the session-level split."""
    from research.advanced_manifest_schema import split_manifest

    split_records = split_manifest(records, seed=20261118)
    counts = {"train": 0, "validation": 0, "test": 0}
    if not all(any(record["split"] == split for record in split_records) for split in counts):
        raise ValueError("split session-level menghasilkan train/validation/test kosong")
    if output.exists():
        shutil.rmtree(output)
    for record in split_records:
        source = image_root / str(record["image"])
        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            x1, y1, x2, y2 = record["bbox"]
            crop = image.crop((x1 * width, y1 * height, x2 * width, y2 * height))
            if crop.width < 2 or crop.height < 2:
                raise ValueError(f"crop terlalu kecil untuk object_id {record['object_id']}")
            split_name = "val" if record["split"] == "validation" else str(record["split"])
            destination = (
                output
                / split_name
                / _safe_name(str(record["label"]))
                / f"{_safe_name(str(record['object_id']))}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop.save(destination, format="JPEG", quality=95)
            counts[str(record["split"])] += 1
    return output, counts


def _blocked(run_dir: Path, task: dict[str, Any], reason: str, manifest: Path) -> dict[str, Any]:
    result = {
        "status": "blocked_no_dataset",
        "task": str(task["task"]),
        "reason": reason,
        "manifest": str(manifest),
        "manifest_sha256": _manifest_hash(manifest),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_status_atomic(run_dir / "status.json", result)
    return result


def run_advanced_task(task: dict[str, Any], manifest: Path, output: Path, device: str, stop_file: Path | None = None) -> dict[str, Any]:
    run_dir = output / str(task["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        return _blocked(run_dir, task, "manifest tidak ditemukan", manifest)
    taxonomy = Path(str(task.get("taxonomy", "research/advanced_taxonomy.yaml")))
    report = validate_manifest(manifest, manifest.parent, taxonomy)
    records = [record for record in report.records if record.get("task") == task.get("task")]
    if not report.valid or not records:
        reason = "manifest tidak valid" if not report.valid else "belum ada record berlabel untuk task"
        return _blocked(run_dir, task, reason, manifest)
    required = {"id", "task", "model"}
    missing = sorted(field for field in required if not str(task.get(field, "")).strip())
    if missing:
        return _blocked(run_dir, task, f"matrix task tidak memiliki field: {', '.join(missing)}", manifest)
    if str(task["task"]) not in TASKS:
        return _blocked(run_dir, task, "task tidak diizinkan oleh taxonomy", manifest)
    dataset_root = run_dir / "dataset"
    try:
        dataset_root, split_counts = _build_task_dataset(records, manifest.parent, dataset_root)
    except (OSError, ValueError) as exc:
        return _blocked(run_dir, task, f"dataset task tidak siap: {exc}", manifest)
    model_path = Path(str(task["model"]))
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    checkpoint = find_valid_checkpoint(run_dir)
    if checkpoint is None and not model_path.is_file():
        return _blocked(run_dir, task, f"bobot awal lokal tidak ditemukan: {model_path}", manifest)
    status = {
        "status": "running",
        "task": str(task["task"]),
        "manifest": str(manifest),
        "manifest_sha256": _manifest_hash(manifest),
        "device": device,
        "dataset": str(dataset_root),
        "split_counts": split_counts,
        "epochs": int(task.get("epochs", 40)),
        "imgsz": int(task.get("imgsz", 224)),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_status_atomic(run_dir / "status.json", status)
    stop_callback: StopFileCallback | None = None
    try:
        from ultralytics import YOLO

        model = YOLO(str(checkpoint or model_path))
        stop_callback = StopFileCallback(stop_file) if stop_file is not None else None
        if stop_callback is not None:
            model.add_callback("on_train_epoch_start", stop_callback)
        if checkpoint:
            model.train(resume=True)
        else:
            model.train(
                data=str(dataset_root),
                epochs=int(task.get("epochs", 40)),
                imgsz=int(task.get("imgsz", 224)),
                batch=int(task.get("batch", 16)),
                patience=int(task.get("patience", 50)),
                workers=int(task.get("workers", 0)),
                device=device,
                project=str(output.resolve()),
                name=str(task["id"]),
                exist_ok=True,
            )
    except KeyboardInterrupt:
        status.update({"status": "interrupted", "finished_at": datetime.now(timezone.utc).isoformat()})
        write_status_atomic(run_dir / "status.json", status)
        raise
    except Exception as exc:
        status.update({"status": "failed", "error": type(exc).__name__, "finished_at": datetime.now(timezone.utc).isoformat()})
        write_status_atomic(run_dir / "status.json", status)
        raise
    final_status = "stopped" if stop_callback is not None and stop_callback.triggered else "completed"
    status.update({"status": final_status, "finished_at": datetime.now(timezone.utc).isoformat()})
    write_status_atomic(run_dir / "status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Loop training kondisi FruitVision yang resumable.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=Path("research/advanced_training_matrix.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/advanced-condition"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-file", type=Path, default=Path("STOP_TRAINING"))
    parser.add_argument("--task", action="append", default=None, help="batasi ke id task tertentu; bisa diulang")
    args = parser.parse_args()
    for task in load_advanced_matrix(args.matrix):
        if args.stop_file.exists():
            print(f"Training dihentikan oleh file: {args.stop_file}")
            break
        if args.task and str(task["id"]) not in args.task:
            continue
        run_advanced_task(task, args.manifest, args.output, args.device, args.stop_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
