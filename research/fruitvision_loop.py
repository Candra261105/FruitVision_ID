from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RUNS = ROOT / "runs"
LOOP_DIR = RUNS / "loop"
ADVANCED = RUNS / "advanced-condition"
MODELS = ROOT / "models"
STOP_FILE = ROOT / "STOP_TRAINING"
LOOP_STOP = ROOT / "STOP_LOOP"

LOOP_DIR.mkdir(parents=True, exist_ok=True)

from research.detector_training import find_best_weight


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (LOOP_DIR / "loop.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def record(entry: dict[str, Any]) -> None:
    entry = {"at": now(), **entry}
    with (LOOP_DIR / "iterations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass
class StageResult:
    ok: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def run_command(args: list[str], log_path: Path, timeout: int | None = None) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("OMP_NUM_THREADS", "4")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n===== {now()} :: {' '.join(args)}\n")
        handle.flush()
        try:
            completed = subprocess.run(
                args,
                cwd=str(ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                check=False,
            )
            return completed.returncode, ""
        except subprocess.TimeoutExpired:
            return 124, f"timeout setelah {timeout} detik"


def tail(path: Path, lines: int = 25) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def task_status(task_id: str) -> dict[str, Any]:
    return read_json(ADVANCED / task_id / "status.json")


def dataset_counts(task_id: str) -> dict[str, int]:
    root = ADVANCED / task_id / "dataset"
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        folder = root / split
        counts[split] = sum(1 for _ in folder.rglob("*.jpg")) if folder.is_dir() else 0
    return counts


# ---------------------------------------------------------------- classification


CLASSIFY = {
    "maturity": {
        "manifest": "data/advanced/tomato_maturity_manifest.jsonl",
        "dataset_label": "ParsaVictor/tomato-ripeness-detection",
        "license": "CC BY 4.0",
        "min_macro_f1": 0.75,
        "init_epochs": 8,
        "ft_epochs": 6,
    },
    "rot": {
        "manifest": "data/advanced/tomato_rot_manifest.jsonl",
        "dataset_label": "ParsaVictor/tomato-ripeness-detection",
        "license": "CC BY 4.0",
        "min_macro_f1": 0.75,
        "init_epochs": 12,
        "ft_epochs": 6,
    },
    "grade": {
        "manifest": "data/advanced/afruitdb_tomato_grade_manifest.jsonl",
        "dataset_label": "Project-AgML/AFruitDB_fruit_grade_classification",
        "license": "CC BY 4.0",
        "min_macro_f1": 0.70,
        "init_epochs": 10,
        "ft_epochs": 8,
    },
    "disease": {
        "manifest": "data/advanced/cucumber_disease_manifest.jsonl",
        "dataset_label": "Project-AgML/cucumber_disease_classification",
        "license": "lihat manifest license_or_permission",
        "min_macro_f1": 0.75,
        "init_epochs": 10,
        "ft_epochs": 8,
    },
}

LEGACY_RUNS = RUNS / "classify" / "runs" / "advanced-condition"


def init_weights(task_id: str) -> Path:
    legacy = LEGACY_RUNS / task_id / "weights" / "best.pt"
    if legacy.is_file():
        return legacy
    return MODELS / "base" / "yolo11n-cls.pt"


def write_matrix(path: Path, task_id: str, model: Path, epochs: int, patience: int) -> Path:
    path.write_text(
        "tasks:\n"
        f"  - id: {task_id}\n"
        f"    task: {task_id}\n"
        f"    model: {model.as_posix()}\n"
        f"    epochs: {epochs}\n"
        f"    patience: {patience}\n"
        "    batch: 64\n"
        "    imgsz: 160\n"
        "    workers: 0\n",
        encoding="utf-8",
    )
    return path


def train_classifier(task_id: str, variant: str, matrix: Path) -> StageResult:
    log_path = LOOP_DIR / f"{task_id}-{variant}.log"
    args = [
        str(PY),
        "-u",
        "-m",
        "research.advanced_train_loop",
        "--manifest",
        CLASSIFY[task_id]["manifest"],
        "--matrix",
        str(matrix),
        "--output",
        str(ADVANCED),
        "--device",
        "cpu",
        "--task",
        variant,
        "--stop-file",
        str(STOP_FILE),
    ]
    code, note = run_command(args, log_path)
    status = task_status(variant)
    if code != 0 or status.get("status") not in {"completed", "stopped"}:
        return StageResult(False, f"training {variant} gagal (exit {code}, status {status.get('status')}). {note}\n{tail(log_path)}")
    counts = dataset_counts(variant)
    if min(counts.values()) == 0:
        return StageResult(False, f"dataset {variant} tidak lengkap: {counts}")
    return StageResult(True, f"training {variant} {status.get('status')}", {"counts": counts, "status": status.get("status")})


def evaluate_classifier(task_id: str, variant: str, imgsz: int) -> StageResult:
    meta = CLASSIFY[task_id]
    weights = ADVANCED / variant / "weights" / "best.pt"
    if not weights.is_file():
        return StageResult(False, f"bobot tidak ada: {weights}")
    test_dir = ADVANCED / variant / "dataset" / "test"
    args = [
        str(PY),
        "-u",
        "-m",
        "research.evaluate_advanced_classifier",
        "--task",
        task_id,
        "--weights",
        str(weights),
        "--test-dir",
        str(test_dir),
        "--manifest",
        meta["manifest"],
        "--dataset",
        meta["dataset_label"],
        "--dataset-license",
        meta["license"],
        "--imgsz",
        str(imgsz),
        "--report",
        str(ADVANCED / variant / "evaluation.json"),
    ]
    log_path = LOOP_DIR / f"{task_id}-{variant}-eval.log"
    code, note = run_command(args, log_path)
    report = read_json(ADVANCED / variant / "evaluation.json")
    metrics = report.get("metrics", {})
    gate = report.get("gate", {})
    evidence = {
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "support": metrics.get("support"),
        "gate": gate.get("status"),
        "reasons": gate.get("reasons"),
        "per_class": {label: round(value.get("f1", 0), 4) for label, value in (metrics.get("per_class") or {}).items()},
    }
    if code != 0:
        return StageResult(False, f"evaluasi gagal (exit {code}). {note}\n{tail(log_path)}", evidence)
    if gate.get("status") != "eligible":
        return StageResult(False, f"gate {task_id} menolak: {gate.get('reasons')}", evidence)
    if (metrics.get("macro_f1") or 0) < meta["min_macro_f1"]:
        return StageResult(False, f"macro-F1 {metrics.get('macro_f1')} < {meta['min_macro_f1']}", evidence)
    return StageResult(True, f"gate eligible macro-F1 {metrics.get('macro_f1'):.3f}", evidence)


def run_classification_target(task_id: str) -> list[dict[str, Any]]:
    meta = CLASSIFY[task_id]
    manifest = ROOT / meta["manifest"]
    iterations: list[dict[str, Any]] = []
    if not manifest.is_file():
        stages = [StageResult(False, f"manifest belum ada: {manifest}")]
    else:
        stages = []
        variant = task_id
        checkpoint = None
        try:
            from research.advanced_train_loop import find_valid_checkpoint

            checkpoint = find_valid_checkpoint(ADVANCED / variant)
        except Exception:
            checkpoint = None
        if checkpoint is not None:
            matrix = ROOT / "research" / "advanced_training_matrix.yaml"
            log(f"{task_id}: lanjutkan checkpoint {checkpoint.name}")
        else:
            weights = init_weights(task_id)
            matrix = write_matrix(LOOP_DIR / f"{task_id}-matrix.yaml", task_id, weights, meta["init_epochs"], 4)
            log(f"{task_id}: mulai dari {weights} untuk {meta['init_epochs']} epoch")
        result = train_classifier(task_id, variant, matrix)
        stages.append(result)
        if result.ok:
            stages.append(evaluate_classifier(task_id, variant, 160))
        if not stages[-1].ok:
            ft_matrix = LOOP_DIR / f"{task_id}-ft-matrix.yaml"
            best = ADVANCED / variant / "weights" / "best.pt"
            if best.is_file():
                ft_matrix.write_text(
                    "tasks:\n"
                    f"  - id: {task_id}-ft\n"
                    f"    task: {task_id}\n"
                    f"    model: {best.as_posix()}\n"
                    f"    epochs: {meta['ft_epochs']}\n"
                    "    patience: 3\n"
                    "    batch: 64\n"
                    "    imgsz: 160\n"
                    "    workers: 0\n",
                    encoding="utf-8",
                )
                stages.append(train_classifier(task_id, f"{task_id}-ft", ft_matrix))
                if stages[-1].ok:
                    stages.append(evaluate_classifier(task_id, f"{task_id}-ft", 160))
    for index, stage in enumerate(stages, start=1):
        entry = {
            "target": task_id,
            "iteration": index,
            "stage": "train" if index == 1 else "evaluate",
            "ok": stage.ok,
            "detail": stage.detail,
            "evidence": stage.evidence,
        }
        record(entry)
        iterations.append(entry)
        log(f"{task_id} iterasi {index}: {'LULUS' if stage.ok else 'GAGAL'} - {stage.detail.splitlines()[0] if stage.detail else ''}")
        if not stage.ok and index == 1 and task_id in {"grade", "disease"}:
            break
    if stages and stages[-1].ok:
        verify = verify_classification_model(task_id)
        entry = {"target": task_id, "iteration": len(iterations) + 1, "stage": "registry", "ok": verify.ok, "detail": verify.detail, "evidence": verify.evidence}
        record(entry)
        iterations.append(entry)
        log(f"{task_id} registry: {'LULUS' if verify.ok else 'GAGAL'} - {verify.detail}")
    return iterations


def verify_classification_model(task_id: str) -> StageResult:
    model = MODELS / f"{task_id}.pt"
    gate = read_json(MODELS / f"{task_id}.gate.json")
    if not model.is_file():
        return StageResult(False, f"models/{task_id}.pt tidak ada")
    if gate.get("status") != "eligible":
        return StageResult(False, f"gate models/{task_id}.gate.json status {gate.get('status')}")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    code = subprocess.run(
        [str(PY), "-c", "from pathlib import Path; from app.services.advanced_model_registry import load_advanced_models; m=load_advanced_models(Path('models')); print({k: v.status for k, v in m.items()})"],
        cwd=str(ROOT),
        env=dict(env, PYTHONPATH=str(ROOT / "backend")),
        capture_output=True,
        text=True,
    )
    if f"'{task_id}': 'eligible'" not in code.stdout:
        return StageResult(False, f"registry runtime tidak eligible: {code.stdout.strip()} {code.stderr.strip()[-400:]}")
    return StageResult(True, "registry runtime eligible")


# ---------------------------------------------------------------------- detector


def detector_gate(metrics: dict[str, Any]) -> bool:
    required_classes = {"cucumber", "tomato", "green beans"}
    overall = metrics.get("map50") or 0.0
    per_class = metrics.get("per_class") or {}
    if overall < 0.55:
        return False
    if set(per_class) != required_classes:
        return False
    return all(value >= 0.40 for value in per_class.values())


def detector_dataset_path() -> Path:
    """Return the audited detector experiment root, not the leaky legacy set."""
    return ROOT / "data" / "processed" / "combined-top3-cc0-experiment"


def train_detector(epochs: int, imgsz: int, batch: int, name: str, data: Path) -> StageResult:
    script = LOOP_DIR / "train_detector_once.py"
    script.write_text(
        "from __future__ import annotations\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from ultralytics import YOLO\n"
        "from research.detector_training import find_best_weight\n"
        "data, weights, out, epochs, imgsz, batch, name = sys.argv[1:8]\n"
        "model = YOLO(weights)\n"
        "model.train(data=data, epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), workers=0,\n"
        "            device='cpu', project=out, name=name, exist_ok=True, patience=4, plots=False,\n"
        "            val=True, seed=0, cache='ram', verbose=True)\n"
        "best = find_best_weight(Path(out), name, Path.cwd())\n"
        "print('BEST', best, best.stat().st_size)\n",
        encoding="utf-8",
    )
    weights = RUNS / "fruitvision-loop" / "yolo11n-light-original" / "weights" / "last.pt"
    if not weights.is_file():
        weights = MODELS / "base" / "yolo11n.pt"
    if not weights.is_file():
        return StageResult(False, f"bobot awal detector tidak ada: {weights}")
    args = [str(PY), "-u", str(script), str(data), str(weights), str(RUNS / "loop-detector"), str(epochs), str(imgsz), str(batch), name]
    log_path = LOOP_DIR / f"detector-{name}.log"
    code, note = run_command(args, log_path)
    try:
        best = find_best_weight(RUNS / "loop-detector", name, ROOT)
    except (FileNotFoundError, FileExistsError) as exc:
        best = None
        resolution_error = str(exc)
    else:
        resolution_error = ""
    if code != 0 or best is None:
        return StageResult(False, f"training detector gagal (exit {code}). {resolution_error}\n{note}\n{tail(log_path)}")
    return StageResult(True, f"detector {name} selesai", {"best": str(best), "size": best.stat().st_size})


def evaluate_detector(name: str, data: Path, imgsz: int) -> StageResult:
    script = LOOP_DIR / "eval_detector_once.py"
    script.write_text(
        "from __future__ import annotations\n"
        "import json, sys\n"
        "from collections import Counter\n"
        "from pathlib import Path\n"
        "from ultralytics import YOLO\n"
        "weights, data, imgsz, out = sys.argv[1:5]\n"
        "model = YOLO(weights)\n"
        "names = model.names\n"
        "metrics = model.val(data=data, split='test', imgsz=int(imgsz), device='cpu', workers=0, plots=False, verbose=False)\n"
        "box = metrics.box\n"
        "per_class = {}\n"
        "ap50 = getattr(box, 'ap50', None)\n"
        "index = list(getattr(box, 'ap_class_index', []))\n"
        "if ap50 is not None and len(ap50) == len(index):\n"
        "    for position, cls in enumerate(index):\n"
        "        per_class[names[int(cls)]] = float(ap50[position])\n"
        "else:\n"
        "    for position, cls in enumerate(index):\n"
        "        per_class[names[int(cls)]] = float(box.maps[position])\n"
        "payload = {'map50': float(box.map50), 'map50_95': float(box.map), 'per_class': per_class,\n"
        "           'precision': float(box.mp), 'recall': float(box.mr)}\n"
        "Path(out).write_text(json.dumps(payload, indent=2), encoding='utf-8')\n"
        "print(json.dumps(payload, indent=2))\n",
        encoding="utf-8",
    )
    out = LOOP_DIR / f"detector-{name}-metrics.json"
    try:
        best = find_best_weight(RUNS / "loop-detector", name, ROOT)
    except (FileNotFoundError, FileExistsError) as exc:
        return StageResult(False, f"bobot evaluasi tidak ditemukan: {exc}")
    args = [str(PY), "-u", str(script), str(best), str(data), str(imgsz), str(out)]
    log_path = LOOP_DIR / f"detector-{name}-eval.log"
    code, note = run_command(args, log_path)
    metrics = read_json(out)
    if code != 0 or not metrics:
        return StageResult(False, f"evaluasi detector gagal (exit {code}). {note}\n{tail(log_path)}")
    if not detector_gate(metrics):
        return StageResult(False, f"detector belum lolos gate: {json.dumps(metrics, ensure_ascii=False)}", metrics)
    MODELS.mkdir(exist_ok=True)
    shutil.copyfile(best, MODELS / "detector.pt")
    (MODELS / "detector.gate.json").write_text(
        json.dumps(
            {
                "status": "eligible",
                "created_at": now(),
                "model": "detector.pt",
                "dataset": "data/processed/combined-top3-final",
                "split": "test holdout bawaan dataset (tanpa augmentasi)",
                "imgsz": imgsz,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return StageResult(True, f"detector eligible mAP50 {metrics['map50']:.3f}", metrics)


def run_detector_target() -> list[dict[str, Any]]:
    data = detector_dataset_path() / "data.yaml"
    attempts = [
        {"epochs": 8, "imgsz": 512, "batch": 8, "name": "detector-512"},
        {"epochs": 6, "imgsz": 448, "batch": 8, "name": "detector-448"},
    ]
    entries: list[dict[str, Any]] = []
    for attempt in attempts:
        if LOOP_STOP.exists() or STOP_FILE.exists():
            entries.append({"target": "detector", "ok": False, "detail": "dihentikan oleh file stop", "evidence": {}})
            record(entries[-1])
            break
        result = train_detector(attempt["epochs"], attempt["imgsz"], attempt["batch"], attempt["name"], data)
        entry = {"target": "detector", "iteration": len(entries) + 1, "stage": "train", "ok": result.ok, "detail": result.detail, "evidence": result.evidence}
        record(entry)
        entries.append(entry)
        log(f"detector train {attempt['name']}: {'LULUS' if result.ok else 'GAGAL'} - {result.detail.splitlines()[0]}")
        if not result.ok:
            continue
        evaluation = evaluate_detector(attempt["name"], data, attempt["imgsz"])
        entry = {"target": "detector", "iteration": len(entries) + 1, "stage": "evaluate", "ok": evaluation.ok, "detail": evaluation.detail, "evidence": evaluation.evidence}
        record(entry)
        entries.append(entry)
        log(f"detector evaluate {attempt['name']}: {'LULUS' if evaluation.ok else 'GAGAL'} - {evaluation.detail.splitlines()[0]}")
        if evaluation.ok:
            break
    return entries


def verify_detector() -> StageResult:
    model = MODELS / "detector.pt"
    gate = read_json(MODELS / "detector.gate.json")
    if not model.is_file():
        return StageResult(False, "models/detector.pt tidak ada")
    if gate.get("status") != "eligible":
        return StageResult(False, f"gate detector {gate.get('status')}")
    script = LOOP_DIR / "verify_detector_once.py"
    script.write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "from pathlib import Path\n"
        "from ultralytics import YOLO\n"
        "root = Path('data/processed/combined-top3-final/images/test')\n"
        "import collections\n"
        "by_class = collections.defaultdict(list)\n"
        "for label in Path('data/processed/combined-top3-final/labels/test').glob('*.txt'):\n"
        "    lines = [line.split() for line in label.read_text().splitlines() if line.strip()]\n"
        "    if not lines:\n"
        "        continue\n"
        "    cls = int(lines[0][0])\n"
        "    image = root / (label.stem + '.jpg')\n"
        "    if image.is_file() and len(by_class[cls]) < 4:\n"
        "        by_class[cls].append(image)\n"
        "model = YOLO('models/detector.pt')\n"
        "result = {}\n"
        "for cls, images in by_class.items():\n"
        "    hits = 0\n"
        "    for image in images:\n"
        "        pred = model.predict(str(image), conf=0.45, imgsz=512, device='cpu', verbose=False)[0]\n"
        "        labels = [int(b.cls) for b in pred.boxes] if pred.boxes is not None else []\n"
        "        hits += 1 if cls in labels else 0\n"
        "    result[model.names[cls]] = {'hits': hits, 'total': len(images)}\n"
        "print(json.dumps(result))\n",
        encoding="utf-8",
    )
    log_path = LOOP_DIR / "detector-verify.log"
    code, note = run_command([str(PY), "-u", str(script)], log_path)
    if code != 0:
        return StageResult(False, f"verifikasi end-to-end detector gagal. {note}\n{tail(log_path)}")
    payload = tail(log_path, 3).strip().splitlines()[-1]
    result = json.loads(payload)
    total = sum(item["total"] for item in result.values())
    hits = sum(item["hits"] for item in result.values())
    if total < 9 or hits / total < 0.6:
        return StageResult(False, f"deteksi nyata lemah: {result}")
    return StageResult(True, f"deteksi nyata {hits}/{total} benar", result)


# ------------------------------------------------------------------- integration


def run_integration() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    checks: list[tuple[str, list[str], Path, int | None]] = [
        ("backend-tests", [str(PY), "-m", "pytest", "backend/tests", "-q"], LOOP_DIR / "backend-tests.log", 1800),
        ("frontend-build", ["cmd", "/c", "npm.cmd", "run", "build"], LOOP_DIR / "frontend-build.log", 1800),
    ]
    for name, args, log_path, timeout in checks:
        cwd = ROOT / "frontend" if name == "frontend-build" else ROOT
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            try:
                completed = subprocess.run(args, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        ok = code == 0
        entry = {"target": "integration", "stage": name, "ok": ok, "detail": "lulus" if ok else f"exit {code}\n{tail(log_path, 12)}", "evidence": {}}
        record(entry)
        entries.append(entry)
        log(f"integrasi {name}: {'LULUS' if ok else 'GAGAL'}")
    summary = {task: read_json(MODELS / f"{task}.gate.json").get("status", "missing") for task in ("maturity", "rot", "grade", "disease")}
    summary["detector"] = read_json(MODELS / "detector.gate.json").get("status", "missing")
    (LOOP_DIR / "model-status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    eligible = [task for task, status in summary.items() if status == "eligible"]
    required = set(summary)
    entry = {
        "target": "integration",
        "stage": "model-status",
        "ok": required <= set(eligible),
        "detail": f"model eligible: {eligible}",
        "evidence": summary,
    }
    record(entry)
    entries.append(entry)
    return entries


TARGETS = {
    "maturity": lambda: run_classification_target("maturity"),
    "rot": lambda: run_classification_target("rot"),
    "grade": lambda: run_classification_target("grade"),
    "disease": lambda: run_classification_target("disease"),
    "detector": run_detector_target,
    "integration": run_integration,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Loop otomatis FruitVision: latih, uji, gerbang, ulangi.")
    parser.add_argument("--only", action="append", choices=sorted(TARGETS), default=None)
    parser.add_argument("--skip", action="append", choices=sorted(TARGETS), default=None)
    args = parser.parse_args()

    selected = args.only or [name for name in ("maturity", "rot", "grade", "disease", "detector", "integration")]
    selected = [name for name in selected if name not in (args.skip or [])]
    log(f"loop mulai untuk {selected}")
    summary: dict[str, Any] = {}
    for name in selected:
        if LOOP_STOP.exists():
            log("STOP_LOOP terdeteksi, berhenti")
            break
        log(f"=== target {name} ===")
        try:
            entries = TARGETS[name]()
        except Exception as exc:  # noqa: BLE001
            record({"target": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]})
            log(f"target {name} error fatal: {type(exc).__name__}: {exc}")
            summary[name] = False
            continue
        summary[name] = bool(entries) and entries[-1]["ok"]
        log(f"=== target {name} -> {'LULUS' if summary[name] else 'BELUM LULUS'} ===")
    log(f"loop selesai: {json.dumps(summary, ensure_ascii=False)}")
    (LOOP_DIR / "last-run.json").write_text(json.dumps({"at": now(), "summary": summary}, indent=2), encoding="utf-8")
    return 0 if all(summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
