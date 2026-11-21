from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "grade": {
        "parquet": Path("data/advanced/source/hf_parquet/grade"),
        "dataset": "Project-AgML/AFruitDB_fruit_grade_classification",
        "license": "CC BY 4.0 (AFruitDB, Mojumdar et al. 2025)",
        "annotator": "afruitdb-graders",
        "task": "grade",
        "fruit": "tomato",
        "filter_column": "fruit_type",
        "filter_value": "tomatoes",
        "label_map": {0: "A", 1: "B", 2: "C"},
        "grade_meaning": "1st-grade->A, 2nd-grade->B, 3rd-grade->C (urutan ordinal dataset, bukan standar pasar)",
        "out_dir": Path("data/advanced/source/crops/cls_afruitdb"),
        "manifest": Path("data/advanced/afruitdb_tomato_grade_manifest.jsonl"),
        "label_version": "afruitdb-2025-ordinal-v1",
    },
    "cucumber_disease": {
        "parquet": Path("data/advanced/source/hf_parquet/cucumber_disease"),
        "dataset": "Project-AgML/cucumber_disease_classification",
        "license": "CC BY 4.0 (artikel Data in Brief 10.1016/j.dib.2023.109320; kartu HF tidak mencantumkan lisensi)",
        "annotator": "cucumber-disease-authors",
        "task": "disease",
        "fruit": "cucumber",
        "filter_column": None,
        "filter_value": None,
        "label_map": {4: "healthy", 2: "belly_rot", 7: "pythium_fruit_rot"},
        "grade_meaning": "kelas buah saja; kelas daun (Anthracnose 0, Fresh_Leaf 5, Downy_Mildew 3) dibuang",
        "out_dir": Path("data/advanced/source/crops/cls_cucumber_disease"),
        "manifest": Path("data/advanced/cucumber_disease_manifest.jsonl"),
        "label_version": "cucumber-disease-fruit-only-v1",
    },
}


def average_hash(image: Image.Image, size: int = 16) -> str:
    small = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(small.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if value >= mean else "0" for value in pixels)
    return f"{int(bits, 2):0{size * size // 4}x}"


def iter_rows(parquet_dir: Path):
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"tidak ada parquet di {parquet_dir}")
    for path in files:
        handle = pq.ParquetFile(path)
        columns = handle.schema_arrow.names
        for batch in handle.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                yield path.name, columns, row


def extract_image(row: dict) -> bytes | None:
    image = row.get("image")
    if isinstance(image, dict):
        return image.get("bytes")
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Konversi shard parquet klasifikasi HF menjadi manifest advanced.")
    parser.add_argument("--spec", choices=sorted(SPECS), required=True)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--limit", type=int, default=0, help="batasi jumlah gambar (uji cepat)")
    args = parser.parse_args()

    spec = SPECS[args.spec]
    parquet_dir = PROJECT_ROOT / spec["parquet"]
    out_dir = PROJECT_ROOT / spec["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    seen_bytes: dict[str, str] = {}
    skipped = Counter()
    rows_seen = 0

    for shard, columns, row in iter_rows(parquet_dir):
        rows_seen += 1
        if spec["filter_column"]:
            value = str(row.get(spec["filter_column"], "")).strip().lower()
            if value != str(spec["filter_value"]).lower():
                skipped["filter"] += 1
                continue
        raw_label = row.get("label")
        if raw_label not in spec["label_map"]:
            skipped["label"] += 1
            continue
        label = spec["label_map"][raw_label]
        payload = extract_image(row)
        if not payload:
            skipped["image"] += 1
            continue
        try:
            with Image.open(io.BytesIO(payload)) as handle:
                image = handle.convert("RGB")
        except Exception:
            skipped["decode"] += 1
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_bytes:
            skipped["duplicate_bytes"] += 1
            continue
        seen_bytes[digest] = label

        index = len(records)
        name = f"{args.spec}_{index:05d}.jpg"
        target = out_dir / name
        image.save(target, format="JPEG", quality=95)
        saved_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        records.append(
            {
                "image": str(target.relative_to(PROJECT_ROOT / "data/advanced")).replace("\\", "/"),
                "object_id": f"{args.spec}-{index:05d}",
                "fruit": spec["fruit"],
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "label": label,
                "task": spec["task"],
                "source": spec["dataset"],
                "session_id": f"{args.spec}-{average_hash(image)}",
                "annotator": spec["annotator"],
                "label_version": spec["label_version"],
                "license_or_permission": spec["license"],
                "_saved_sha256": saved_hash,
                "_shard": shard,
            }
        )
        if args.limit and len(records) >= args.limit:
            break
        if len(records) % 500 == 0:
            print(f"{len(records)} gambar tersimpan (baris dipindai {rows_seen})", flush=True)

    if not records:
        raise SystemExit("tidak ada record yang lolos filter")

    # dedupe berdasarkan byte file tersimpan
    unique = []
    seen_saved: dict[str, str] = {}
    for record in records:
        digest = record.pop("_saved_sha256")
        if digest in seen_saved:
            skipped["duplicate_saved"] += 1
            (PROJECT_ROOT / "data/advanced" / record["image"]).unlink(missing_ok=True)
            continue
        seen_saved[digest] = record["object_id"]
        unique.append(record)

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in unique:
        groups[record["session_id"]].append(record)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    assignments: dict[str, str] = {}
    for position, key in enumerate(keys):
        share = position / max(len(keys), 1)
        assignments[key] = "test" if share < 0.15 else "validation" if share < 0.30 else "train"

    per_split = Counter()
    per_split_label = Counter()
    for record in unique:
        split = assignments[record["session_id"]]
        record["split"] = split
        per_split[split] += 1
        per_split_label[f"{split}:{record['label']}"] += 1

    manifest = PROJECT_ROOT / spec["manifest"]
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in unique:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    audit = {
        "spec": args.spec,
        "dataset": spec["dataset"],
        "license": spec["license"],
        "task": spec["task"],
        "label_map": {str(key): value for key, value in spec["label_map"].items()},
        "note": spec["grade_meaning"],
        "rows_seen": rows_seen,
        "images_saved": len(unique),
        "skipped": dict(skipped),
        "groups": len(keys),
        "per_split": dict(per_split),
        "per_split_label": dict(per_split_label),
        "label_counts": dict(Counter(record["label"] for record in unique)),
        "manifest": str(manifest.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    audit_path = manifest.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
