from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def validate_source_entry(entry: dict[str, Any], allow_unknown_license: bool = False) -> dict[str, Any]:
    license_name = str(entry.get("license", "")).strip().lower()
    if not license_name or (license_name in {"unknown", "", "license-review-required"} and not allow_unknown_license):
        raise ValueError(f"lisensi tidak jelas untuk sumber {entry.get('id', '<unknown>')}")
    annotation = str(entry.get("annotation", "")).lower()
    if annotation in {"image-folder", "classification", "classification-only"} or "object" not in annotation and "bounding" not in annotation and "segmentation" not in annotation:
        raise ValueError(f"sumber {entry.get('id', '<unknown>')} adalah classification-only atau tidak memiliki anotasi object detection")
    if not entry.get("url"):
        raise ValueError(f"URL unduhan tidak ada untuk sumber {entry.get('id', '<unknown>')}")
    return entry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"} if existing else {})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            supports_resume = response.status == 206
            if existing and not supports_resume:
                existing = 0
                partial.unlink(missing_ok=True)
                with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as fresh:
                    with partial.open("wb") as stream:
                        shutil.copyfileobj(fresh, stream)
            else:
                mode = "ab" if existing else "wb"
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream)
    except Exception:
        raise
    partial.replace(target)
    return {"url": url, "path": str(target), "size": target.stat().st_size, "sha256": sha256_file(target), "status": "downloaded"}


def download_source(entry: dict[str, Any], output: Path, allow_unknown_license: bool = False) -> dict[str, Any]:
    validate_source_entry(entry, allow_unknown_license=allow_unknown_license)
    target = output / str(entry["id"]) / str(entry.get("filename", "source.bin"))
    result = download_file(str(entry["url"]), target)
    result.update({"id": entry["id"], "license": entry["license"], "annotation": entry["annotation"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Unduh sumber object-detection dengan resume dan checksum.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", dest="source_ids")
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("research/reports/downloads.json"))
    args = parser.parse_args()
    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    selected = [entry for entry in entries if not args.source_ids or entry.get("id") in args.source_ids]
    if not selected:
        raise SystemExit("Tidak ada sumber yang dipilih.")
    results = [download_source(entry, args.output, args.allow_unknown_license) for entry in selected]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
