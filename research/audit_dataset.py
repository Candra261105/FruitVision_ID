from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_class_name(name: str) -> str:
    """Normalize obvious spelling/plural variants without merging fruit types."""
    value = re.sub(r"[_-]+", " ", str(name).strip().lower())
    value = re.sub(r"\s+", " ", value)
    plural_map = {
        "apples": "apple",
        "bananas": "banana",
        "carrots": "carrot",
        "cucumbers": "cucumber",
        "green bean": "green beans",
        "green beans": "green beans",
        "lemons": "lemon",
        "oranges": "orange",
        "potatoes": "potato",
        "tomates": "tomato",
        "tomatoes": "tomato",
        "tomat": "tomato",
    }
    return plural_map.get(value, value)


def _image_files(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _find_coco_file(source: Path) -> Path | None:
    for path in sorted(source.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and {"images", "categories", "annotations"} <= payload.keys():
            return path
    return None


def _resolve_image(source: Path, file_name: str) -> Path | None:
    candidate = source / file_name
    if candidate.is_file():
        return candidate
    matches = list(source.rglob(Path(file_name).name))
    return matches[0] if len(matches) == 1 else None


def _valid_coco_box(box: Any, width: Any, height: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x, y, box_width, box_height = (float(value) for value in box)
        image_width, image_height = float(width), float(height)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height, image_width, image_height)):
        return False
    if box_width <= 0 or box_height <= 0 or image_width <= 0 or image_height <= 0:
        return False
    return x < image_width and y < image_height and x + box_width > 0 and y + box_height > 0


def _audit_coco(source: Path, annotation_path: Path) -> dict[str, Any]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: str(item["name"]) for item in payload.get("categories", []) if "id" in item and "name" in item}
    images = {item["id"]: item for item in payload.get("images", []) if "id" in item for item in [item]}
    counts: Counter[str] = Counter()
    image_sets: dict[str, set[str]] = defaultdict(set)
    counts_by_image: dict[str, Counter[str]] = defaultdict(Counter)
    invalid = 0
    unresolved = 0
    original_names: dict[str, Counter[str]] = defaultdict(Counter)
    for annotation in payload.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        raw_name = categories.get(annotation.get("category_id"))
        if image is None or raw_name is None:
            invalid += 1
            continue
        if not _valid_coco_box(annotation.get("bbox"), image.get("width"), image.get("height")):
            invalid += 1
            continue
        image_path = _resolve_image(source, str(image.get("file_name", "")))
        if image_path is None:
            unresolved += 1
            continue
        name = normalize_class_name(raw_name)
        counts[name] += 1
        image_hash = sha256_file(image_path)
        image_sets[name].add(image_hash)
        counts_by_image[name][image_hash] += 1
        original_names[name][raw_name] += 1
    return {
        "annotation_format": "coco",
        "annotation_file": str(annotation_path.relative_to(source)),
        "valid_box_counts": dict(sorted(counts.items())),
        "valid_image_counts": {name: len(hashes) for name, hashes in sorted(image_sets.items())},
        "valid_image_hashes": {name: sorted(hashes) for name, hashes in sorted(image_sets.items())},
        "valid_box_counts_by_image": {name: dict(values) for name, values in sorted(counts_by_image.items())},
        "original_class_names": {name: dict(names) for name, names in sorted(original_names.items())},
        "invalid_box_count": invalid,
        "unresolved_image_count": unresolved,
    }


def _find_yolo_names(source: Path) -> dict[int, str] | None:
    for path in sorted(source.rglob("*.yaml")) + sorted(source.rglob("*.yml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        names = payload.get("names")
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
    return None


def _audit_yolo(source: Path) -> dict[str, Any] | None:
    names = _find_yolo_names(source)
    if names is None:
        return None
    counts: Counter[str] = Counter()
    image_sets: dict[str, set[str]] = defaultdict(set)
    counts_by_image: dict[str, Counter[str]] = defaultdict(Counter)
    invalid = 0
    for image_path in _image_files(source):
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            try:
                relative_image = image_path.relative_to(source / "images")
            except ValueError:
                relative_image = None
            if relative_image is not None:
                label_path = (source / "labels" / relative_image).with_suffix(".txt")
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                invalid += 1
                continue
            if class_id not in names or len(values) != 4 or not all(math.isfinite(value) for value in values):
                invalid += 1
                continue
            _, _, width, height = values
            if not all(0.0 <= value <= 1.0 for value in values) or width <= 0 or height <= 0:
                invalid += 1
                continue
            name = normalize_class_name(names[class_id])
            counts[name] += 1
            image_hash = sha256_file(image_path)
            image_sets[name].add(image_hash)
            counts_by_image[name][image_hash] += 1
    return {
        "annotation_format": "yolo",
        "valid_box_counts": dict(sorted(counts.items())),
        "valid_image_counts": {name: len(hashes) for name, hashes in sorted(image_sets.items())},
        "valid_image_hashes": {name: sorted(hashes) for name, hashes in sorted(image_sets.items())},
        "valid_box_counts_by_image": {name: dict(values) for name, values in sorted(counts_by_image.items())},
        "invalid_box_count": invalid,
    }


def audit_source(source: Path) -> dict[str, Any]:
    if not source.exists():
        return {"source": str(source), "exists": False, "image_count": 0, "classes": {}, "files": []}
    files = _image_files(source)
    hashes = [sha256_file(path) for path in files]
    result: dict[str, Any] = {
        "source": str(source),
        "exists": True,
        "image_count": len(files),
        "unique_image_count": len(set(hashes)),
        "duplicate_image_count": len(files) - len(set(hashes)),
        "classes": dict(sorted(Counter(path.parent.name for path in files).items())),
        "files": [{"path": str(path.relative_to(source)), "sha256": digest} for path, digest in zip(files[:1000], hashes[:1000])],
        "truncated_file_hashes": len(files) > 1000,
    }
    annotation_path = _find_coco_file(source)
    annotation_audit = _audit_coco(source, annotation_path) if annotation_path else _audit_yolo(source)
    if annotation_audit:
        result.update(annotation_audit)
    else:
        result["annotation_format"] = "unknown"
        result["valid_box_counts"] = {}
        result["valid_image_counts"] = {}
        result["valid_image_hashes"] = {}
        result["valid_box_counts_by_image"] = {}
        result["invalid_box_count"] = 0
    return result


def aggregate_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    box_counts: Counter[str] = Counter()
    image_hashes: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    box_counts_by_image: dict[str, dict[str, int]] = defaultdict(dict)
    for audit in audits:
        source = str(audit.get("source", ""))
        for name, count in audit.get("valid_box_counts", {}).items():
            sources[name].add(source)
        for name, hashes in audit.get("valid_image_hashes", {}).items():
            image_hashes[name].update(hashes)
        for name, image_counts in audit.get("valid_box_counts_by_image", {}).items():
            for image_hash, count in image_counts.items():
                box_counts_by_image[name][image_hash] = max(box_counts_by_image[name].get(image_hash, 0), int(count))
        for name, count in audit.get("valid_box_counts", {}).items():
            if name not in audit.get("valid_box_counts_by_image", {}):
                box_counts[name] += int(count)
                image_hashes[name].update(f"{source}:{index}" for index in range(int(audit.get("valid_image_counts", {}).get(name, 0))))
    for name, image_counts in box_counts_by_image.items():
        box_counts[name] = sum(image_counts.values())
    ranking = sorted(
        (
            {
                "class_name": name,
                "valid_box_count": count,
                "unique_image_count": len(image_hashes[name]),
                "source_count": len(sources[name]),
            }
            for name, count in box_counts.items()
        ),
        key=lambda item: (-item["valid_box_count"], -item["unique_image_count"], item["class_name"]),
    )
    return {"classes": ranking, "top3": ranking[:3]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit metadata dan anotasi dataset tanpa mengunduh data.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", help="Folder dataset lokal. Ulangi opsi ini untuk agregasi lintas sumber.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "manifest": str(args.manifest),
        "schema_version": manifest.get("schema_version"),
        "entry_count": len(manifest.get("entries", [])),
        "entries": manifest.get("entries", []),
    }
    if args.source:
        audits = [audit_source(source.resolve()) for source in args.source]
        if len(audits) == 1:
            result["local_source_audit"] = audits[0]
        else:
            result["local_source_audits"] = audits
            result["aggregate"] = aggregate_audits(audits)
    output = args.output or args.manifest.with_suffix(".audit.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit selesai: {output}")
    print(f"Entri manifest: {result['entry_count']}")
    if args.source:
        if len(args.source) == 1:
            print(f"Gambar lokal: {result['local_source_audit']['image_count']}")
        else:
            print(f"Sumber lokal diaudit: {len(args.source)}")
            print("Top 3:")
            for item in result["aggregate"]["top3"]:
                print(f"- {item['class_name']}: {item['valid_box_count']} box valid")
    print("Tidak ada dataset yang diunduh atau dihapus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
