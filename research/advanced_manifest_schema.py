from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = {
    "image", "object_id", "fruit", "bbox", "label", "task", "source",
    "session_id", "annotator", "label_version", "license_or_permission",
}


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"baris {line_number} bukan JSON valid") from exc
        if not isinstance(record, dict):
            raise ValueError(f"baris {line_number} harus berupa object")
        record["_line"] = line_number
        records.append(record)
    return records


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest_path: Path, image_root: Path, taxonomy_path: Path) -> ValidationReport:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    try:
        records = _load_records(manifest_path)
        taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return ValidationReport(False, [str(exc)])
    fruits = set(taxonomy.get("fruits", ["cucumber", "tomato", "green_beans"]))
    tasks = taxonomy.get("tasks", {})
    hashes: dict[str, str] = {}
    seen_objects: set[str] = set()
    valid_records: list[dict[str, Any]] = []
    for record in records:
        line = record.get("_line", "?")
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            errors.append(f"baris {line}: field kurang {sorted(missing)}")
            continue
        image_path = image_root / str(record["image"])
        bbox = record["bbox"]
        task = str(record["task"])
        label = str(record["label"])
        if not image_path.is_file():
            errors.append(f"baris {line}: image tidak ditemukan")
        if record["fruit"] not in fruits:
            errors.append(f"baris {line}: fruit tidak didukung")
        if task not in tasks or label not in tasks.get(task, []):
            errors.append(f"baris {line}: task/label tidak ada di taxonomy")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(value, (int, float)) or value < 0 or value > 1 for value in bbox):
            errors.append(f"baris {line}: bbox harus empat angka dalam rentang 0..1")
        elif bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            errors.append(f"baris {line}: bbox tidak memiliki luas positif")
        for field_name in ("source", "session_id", "annotator", "label_version", "license_or_permission"):
            if not str(record[field_name]).strip():
                errors.append(f"baris {line}: {field_name} kosong")
        object_id = str(record["object_id"])
        if object_id in seen_objects:
            errors.append(f"baris {line}: object_id duplikat")
        seen_objects.add(object_id)
        if image_path.is_file():
            digest = _file_hash(image_path)
            if digest in hashes:
                errors.append(f"baris {line}: image exact duplicate dengan {hashes[digest]}")
            hashes[digest] = str(record["image"])
        if not any(error.startswith(f"baris {line}:") for error in errors):
            valid_records.append(record)
            counts[f"{task}:{label}"] += 1
    return ValidationReport(not errors, errors, dict(counts), valid_records)


PINNED_SPLITS = {"train", "validation", "test"}


def split_manifest(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    pinned = [str(record.get("split", "")).strip() for record in records]
    if any(pinned):
        if any(value not in PINNED_SPLITS for value in pinned):
            raise ValueError("split record harus train, validation, atau test")
        assigned: dict[tuple[str, str], str] = {}
        for record, split in zip(records, pinned):
            key = (str(record["source"]), str(record["session_id"]))
            if key in assigned and assigned[key] != split:
                raise ValueError(f"session {key[1]} muncul di lebih dari satu split")
            assigned[key] = split
        return [dict(record, split=split) for record, split in zip(records, pinned)]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["source"]), str(record["session_id"]))].append(record)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    total = len(keys)
    assignments: dict[tuple[str, str], str] = {}
    for index, key in enumerate(keys):
        fraction = index / max(total, 1)
        assignments[key] = "test" if fraction < 0.15 else "validation" if fraction < 0.30 else "train"
    result = []
    for record in records:
        copy = dict(record)
        copy["split"] = assignments[(str(record["source"]), str(record["session_id"]))]
        result.append(copy)
    return result
