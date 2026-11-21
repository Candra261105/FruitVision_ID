"""Susun manifest maturity & rot dari crop tomat ParsaVictor (CC BY 4.0).

Sumber: https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection
Struktur: cls_tomato/{train,val,test}/{raw,half_ripe,ripe,rotten}/*.jpg (160x160).

Split memakai pembagian bawaan dataset setelah diaudit: tidak ada satu gambar asal
(session) yang muncul di dua split. Pemilihan sampel dibatasi per sumber agar satu
sumber tidak mendominasi, dan augmentasi bawaan hanya dipakai di train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROP_ROOT = PROJECT_ROOT / "data" / "advanced" / "source" / "crops" / "cls_tomato"
SOURCE_NAME = "parsavictor/tomato-ripeness-detection"
SOURCE_URL = "https://huggingface.co/datasets/ParsaVictor/tomato-ripeness-detection"
DATASET_REPO_SHA = "91481266b3674f4eec31e72310888321d0e102f2"
CROPS_ZIP_SHA256 = "39b163cb1ffcadc8938fd7fd5ee560561b543be7698079ad6d6299159b951909"
LICENSE = "CC BY 4.0"
SESSION_RE = re.compile(r"^(?P<session>.+)_(?P<index>\d+)$")
SPLIT_DIRS = ("train", "val", "test")
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}
MATURITY_LABELS = {"raw": "unripe", "half_ripe": "half_ripe", "ripe": "ripe"}
ROT_LABELS = {
    "raw": "not_rotten",
    "half_ripe": "not_rotten",
    "ripe": "not_rotten",
    "rotten": "rotten",
}
TARGETS = {
    "maturity": {
        "train": {"unripe": 3000, "half_ripe": 3000, "ripe": 3000},
        "val": {"unripe": 350, "half_ripe": 350, "ripe": 350},
        "test": {"unripe": None, "half_ripe": None, "ripe": None},
    },
    "rot": {
        "train": {"not_rotten": 3000, "rotten": 3000},
        "val": {"not_rotten": 200, "rotten": 200},
        "test": {"not_rotten": None, "rotten": None},
    },
}
MAX_SOURCE_SHARE = {"maturity": 0.40, "rot": 0.50}


def session_of(path: Path) -> str:
    match = SESSION_RE.match(path.stem)
    return match.group("session") if match else path.stem


def source_of(path: Path) -> str:
    return path.stem.split("__", 1)[0]


def collect_files() -> dict[str, dict[str, list[Path]]]:
    inventory: dict[str, dict[str, list[Path]]] = {}
    for split in SPLIT_DIRS:
        inventory[split] = {}
        for label_dir in sorted((CROP_ROOT / split).iterdir()):
            if label_dir.is_dir():
                inventory[split][label_dir.name] = sorted(label_dir.glob("*.jpg"))
    return inventory


def dedupe_inventory(inventory: dict[str, dict[str, list[Path]]]) -> list[dict[str, str]]:
    """Buang file dengan isi identik; validator menolak duplikat exact."""
    seen: dict[str, Path] = {}
    dropped: list[dict[str, str]] = []
    for split in SPLIT_DIRS:
        for label in sorted(inventory[split]):
            kept: list[Path] = []
            for path in inventory[split][label]:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    dropped.append({"file": path.relative_to(PROJECT_ROOT).as_posix(), "duplicate_of": seen[digest].relative_to(PROJECT_ROOT).as_posix()})
                    continue
                seen[digest] = path
                kept.append(path)
            inventory[split][label] = kept
    return dropped


def allocate(sizes: dict[str, int], total: int, max_share: float) -> dict[str, int]:
    """Bagi kuota proporsional dengan batas maksimum per sumber."""
    available = sum(sizes.values())
    if total >= available:
        return dict(sizes)
    caps = {name: min(size, max(1, math.ceil(total * max_share))) for name, size in sizes.items()}
    exact = {name: total * size / available for name, size in sizes.items()}
    quota = {name: min(caps[name], math.floor(exact[name])) for name in sizes}
    remainder = total - sum(quota.values())
    while remainder > 0:
        candidates = [name for name in sizes if quota[name] < caps[name]]
        if not candidates:
            break
        candidates.sort(
            key=lambda name: (exact[name] - math.floor(exact[name]), sizes[name] - quota[name]),
            reverse=True,
        )
        quota[candidates[0]] += 1
        remainder -= 1
    return quota


def sample_paths(paths: list[Path], target: int | None, rng: random.Random, max_share: float) -> list[Path]:
    if target is None or target >= len(paths):
        return list(paths)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[source_of(path)].append(path)
    quota = allocate({name: len(group) for name, group in grouped.items()}, target, max_share)
    chosen: list[Path] = []
    for name in sorted(grouped):
        group = grouped[name]
        take = min(quota.get(name, 0), len(group))
        chosen.extend(rng.sample(group, take))
    return sorted(chosen)


def build_task_rows(task: str, inventory: dict[str, dict[str, list[Path]]], seed: int, relative_root: Path) -> list[dict[str, Any]]:
    label_map = MATURITY_LABELS if task == "maturity" else ROT_LABELS
    rng = random.Random(f"{seed}-{task}")
    rows: list[dict[str, Any]] = []
    for split in SPLIT_DIRS:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for source_label, paths in inventory[split].items():
            mapped = label_map.get(source_label)
            if mapped is not None:
                grouped[mapped].extend(paths)
        chosen: dict[str, list[Path]] = {}
        for label in sorted(grouped):
            target = TARGETS[task][split].get(label)
            chosen[label] = sample_paths(grouped[label], target, rng, MAX_SOURCE_SHARE[task])
        if task == "rot" and split != "test":
            common = min(len(paths) for paths in chosen.values())
            for label in chosen:
                if len(chosen[label]) > common:
                    chosen[label] = rng.sample(sorted(chosen[label]), common)
        for label in sorted(chosen):
            for path in chosen[label]:
                relative = path.relative_to(relative_root).as_posix()
                rows.append(
                    {
                        "image": relative,
                        "object_id": f"{task}-{split}-{label}-{path.stem}",
                        "fruit": "tomato",
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "label": label,
                        "task": task,
                        "source": SOURCE_NAME,
                        "session_id": session_of(path),
                        "annotator": "parsavictor-dataset",
                        "label_version": "crops-2026-10-27",
                        "license_or_permission": f"{LICENSE} (ParsaVictor/tomato-ripeness-detection)",
                        "split": SPLIT_NAMES[split],
                    }
                )
    rows.sort(key=lambda row: (row["split"], row["label"], row["image"]))
    return rows


def audit(inventory: dict[str, dict[str, list[Path]]], rows_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    sessions: dict[str, set[str]] = {split: set() for split in SPLIT_DIRS}
    source_counts: dict[str, Counter[str]] = {}
    for split in SPLIT_DIRS:
        counter: Counter[str] = Counter()
        for label, paths in inventory[split].items():
            for path in paths:
                counter[f"{label}:{source_of(path)}"] += 1
                sessions[split].add(session_of(path))
        source_counts[split] = counter
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left}_{right}"] = len(sessions[left] & sessions[right])

    selection: dict[str, Any] = {}
    for task, rows in rows_by_task.items():
        counts: Counter[str] = Counter()
        prefixes: Counter[str] = Counter()
        split_sessions: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            counts[f"{row['split']}:{row['label']}"] += 1
            prefixes[f"{row['split']}:{source_of(Path(row['image']))}"] += 1
            split_sessions[row["split"]].add(row["session_id"])
        leaked = (
            len(split_sessions["train"] & split_sessions["validation"])
            + len(split_sessions["train"] & split_sessions["test"])
            + len(split_sessions["validation"] & split_sessions["test"])
        )
        selection[task] = {
            "counts": dict(sorted(counts.items())),
            "source_mix": dict(sorted(prefixes.items())),
            "sessions": {split: len(values) for split, values in sorted(split_sessions.items())},
            "session_overlap": leaked,
        }

    samples = []
    for split in SPLIT_DIRS:
        for label in sorted(inventory[split]):
            paths = inventory[split][label]
            for path in paths[:: max(1, len(paths) // 12)][:12]:
                with Image.open(path) as image:
                    samples.append({"file": path.name, "size": list(image.size), "mode": image.mode})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo": SOURCE_NAME,
            "url": SOURCE_URL,
            "repo_sha": DATASET_REPO_SHA,
            "crops_zip_etag_sha256": CROPS_ZIP_SHA256,
            "license": LICENSE,
            "crop_size": [160, 160],
        },
        "inventory_files": {split: sum(len(paths) for paths in inventory[split].values()) for split in SPLIT_DIRS},
        "inventory_sessions": {split: len(values) for split, values in sessions.items()},
        "inventory_source_mix": {split: dict(sorted(counter.items())) for split, counter in source_counts.items()},
        "session_overlap": overlaps,
        "selection": selection,
        "image_samples": samples,
        "notes": [
            "Split train/validation/test mengikuti pembagian bawaan dataset; audit menunjukkan 0 session lintas split.",
            "Test set seluruhnya berasal dari sumber def__ (Tomato Defects), sehingga metrik hanya mewakili domain tersebut.",
            "Kelas rotten dilatih juga dari varian flip/rotasi bawaan dataset; evaluasi tidak memakai augmentasi.",
            "maturity tidak memakai kelas rotten; rotten dinilai oleh task rot terpisah.",
            "overripe tidak tersedia sebagai label sumber, jadi tidak dipaksakan ke taxonomy.",
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Buat manifest maturity/rot dari crop tomat.")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "advanced")
    args = parser.parse_args()

    inventory = collect_files()
    if not inventory["train"]:
        raise SystemExit(f"crop tidak ditemukan di {CROP_ROOT}")
    dropped = dedupe_inventory(inventory)

    rows_by_task = {task: build_task_rows(task, inventory, args.seed, args.output_dir) for task in ("maturity", "rot")}
    for task, rows in rows_by_task.items():
        path = args.output_dir / f"tomato_{task}_manifest.jsonl"
        write_jsonl(path, rows)
        print(f"{task}: {len(rows)} record -> {path} ({sha256_file(path)[:12]})")

    report = audit(inventory, rows_by_task)
    report["dropped_exact_duplicates"] = dropped
    report_path = args.output_dir / "tomato_crops_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"audit -> {report_path}")
    for task, payload in report["selection"].items():
        print(task, json.dumps(payload["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
