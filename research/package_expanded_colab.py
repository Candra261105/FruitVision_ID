"""Package expanded dataset and Colab training runner into zip archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def package_dataset_for_colab(
    dataset_root: Path,
    training_script: Path,
    output_zip: Path,
) -> Path:
    """Create a self-contained zip package for Google Colab training."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        # Add dataset files
        for file_path in dataset_root.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(dataset_root)
                zf.write(file_path, arcname=f"dataset/{rel_path}".replace("\\", "/"))

        # Add training script
        if training_script.is_file():
            zf.write(training_script, arcname=f"training/{training_script.name}")

    return output_zip


def main() -> int:
    parser = argparse.ArgumentParser(description="Package expanded dataset for Colab.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/tomato-pumpkin-lemon-expanded/dataset"),
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("data/tomato-pumpkin-lemon-expanded/training/colab_train_expanded.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tomato-pumpkin-lemon-expanded.zip"),
    )
    args = parser.parse_args()

    print(f"[*] Packaging expanded dataset from {args.dataset}...")
    zip_path = package_dataset_for_colab(args.dataset, args.script, args.output)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Package ready: {zip_path} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
