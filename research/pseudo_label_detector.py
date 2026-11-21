from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TARGET_CLASSES = ("cucumber", "tomato", "green beans")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_candidate(
    image: Path,
    detections: list[dict[str, Any]],
    output: Path,
    *,
    model: str,
    source: str,
    min_confidence: float,
) -> bool:
    """Write one AI proposal to a review-only manifest.

    This deliberately does not write YOLO labels. A human audit must promote
    a proposal before it can enter a training split.
    """
    valid = [
        item
        for item in detections
        if str(item.get("class_name", "")) in TARGET_CLASSES
        and float(item.get("confidence", 0.0)) >= min_confidence
    ]
    if not valid:
        return False
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "image": str(image),
        "sha256": sha256_file(image),
        "source": source,
        "model": model,
        "status": "review",
        "detections": valid,
    }
    with (output / "candidates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Buat pseudo-label detector sebagai kandidat review.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    args = parser.parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence harus berada pada 0..1")

    # Model invocation is intentionally supplied by the caller in a later
    # stage. The module's safe contract is the manifest writer above.
    count = 0
    for image in sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        count += int(write_candidate(image, [], args.output, model=args.model, source=args.source, min_confidence=args.min_confidence))
    print(json.dumps({"written": count, "status": "review_only"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
