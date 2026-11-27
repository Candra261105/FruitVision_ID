"""Downloader utility for FruitVision ID production 35-class fruit & vegetable model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

MODEL_REPO = "Piyu12/fruit-veg-yolo11m-detector"
MODEL_FILENAME = "fruit_veg_yolo11m.pt"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/best.pt"


def ensure_production_model(destination: Path | None = None) -> Path:
    """Check if model exists locally; if not, download it from Hugging Face."""
    if destination is None:
        root = Path(__file__).resolve().parent.parent
        destination = root / "models" / MODEL_FILENAME

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size > 1024 * 1024:
        return destination

    print(f"[*] Mengunduh model produksi 35 kelas dari Hugging Face ({MODEL_REPO})...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "FruitVisionID/1.0"})
    with urllib.request.urlopen(req) as resp, destination.open("wb") as f:
        while chunk := resp.read(1024 * 512):
            f.write(chunk)

    print(f"[OK] Model berhasil disimpan ke {destination} ({destination.stat().st_size / (1024*1024):.1f} MB)")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Unduh model produksi FruitVision ID dari Hugging Face")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/fruit_veg_yolo11m.pt"),
        help="Path target berkas bobot .pt",
    )
    args = parser.parse_args()
    try:
        ensure_production_model(args.output)
        return 0
    except Exception as err:
        print(f"[ERROR] Gagal mengunduh model: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
