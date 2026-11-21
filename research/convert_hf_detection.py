from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

from research.audit_dataset import normalize_class_name


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise SystemExit("pyarrow belum terpasang. Jalankan: uv pip install --python .venv\\Scripts\\python.exe pyarrow") from exc
    return parquet


def _bytes(value: Any) -> bytes:
    if isinstance(value, dict) and "bytes" in value:
        return value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError("kolom image tidak berisi bytes yang dikenali")


def read_parquet_class_names(source: Path) -> dict[int, str]:
    """Read Hugging Face ClassLabel names without depending on a dataset SDK."""
    parquet = _require_pyarrow()
    metadata = parquet.ParquetFile(source).schema_arrow.metadata or {}
    raw = metadata.get(b"huggingface")
    if not raw:
        raise ValueError(f"metadata Hugging Face tidak ditemukan: {source}")
    payload = json.loads(raw.decode("utf-8"))
    names = payload["info"]["features"]["objects"]["categories"]["feature"]["names"]
    return {index: str(name) for index, name in enumerate(names)}


def convert_parquet(source: Path, output: Path, dataset_id: str, class_names: dict[int, str]) -> dict[str, int]:
    parquet = _require_pyarrow()
    table = parquet.read_table(source)
    rows = table.to_pylist()
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    counts: dict[str, int] = {}
    normalized_names: list[str] = []
    category_to_label: dict[int, int] = {}
    for category_id, name in class_names.items():
        normalized = normalize_class_name(name)
        if normalized not in normalized_names:
            normalized_names.append(normalized)
        category_to_label[category_id] = normalized_names.index(normalized)
    for index, row in enumerate(rows):
        image = row.get("image")
        objects = row.get("objects") or {}
        raw_boxes = objects.get("bbox") or []
        categories = objects.get("categories") or []
        image_bytes = _bytes(image)
        image_name = f"{dataset_id}_{index:06d}.jpg"
        (images_dir / image_name).write_bytes(image_bytes)
        label_lines: list[str] = []
        for box, category in zip(raw_boxes, categories):
            if isinstance(category, dict):
                category_id = int(category.get("id", category.get("label", -1)))
            else:
                category_id = int(category)
            if category_id not in class_names:
                continue
            values = [float(value) for value in box]
            if len(values) != 4:
                continue
            x, y, width, height = values
            image_width = float(image.get("width", 0)) if isinstance(image, dict) else 0
            image_height = float(image.get("height", 0)) if isinstance(image, dict) else 0
            if image_width <= 0 or image_height <= 0:
                try:
                    from PIL import Image
                    with Image.open(io.BytesIO(image_bytes)) as decoded:
                        image_width, image_height = decoded.size
                except Exception:
                    image_width = image_height = 0
            if image_width <= 0 or image_height <= 0:
                continue
            label_id = category_to_label[category_id]
            label_lines.append(f"{label_id} {(x + width / 2) / image_width:.8f} {(y + height / 2) / image_height:.8f} {width / image_width:.8f} {height / image_height:.8f}")
            name = normalized_names[label_id]
            counts[name] = counts.get(name, 0) + 1
        (labels_dir / image_name.replace(".jpg", ".txt")).write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    (output / "classes.json").write_text(json.dumps(normalized_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data.yaml").write_text(
        json.dumps({"path": str(output.resolve()), "train": "images", "val": "images", "names": normalized_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Konversi Parquet object detection Hugging Face ke folder YOLO.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True, help="JSON mapping category id ke nama kelas.")
    args = parser.parse_args()
    classes = {int(key): str(value) for key, value in json.loads(args.classes.read_text(encoding="utf-8")).items()}
    total: dict[str, int] = {}
    for source in args.source:
        counts = convert_parquet(source, args.output, source.stem, classes)
        for name, count in counts.items():
            total[name] = total.get(name, 0) + count
    print(json.dumps(total, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
