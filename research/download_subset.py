from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Salin subset lokal yang sudah diunduh secara eksplisit.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Batas file per kelas. 0 berarti semua file.")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Folder sumber tidak ditemukan: {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)
    copied = 0
    for class_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        target = args.output / class_dir.name
        target.mkdir(exist_ok=True)
        files = sorted(path for path in class_dir.iterdir() if path.is_file())
        selected = files[: args.limit] if args.limit > 0 else files
        for file in selected:
            shutil.copy2(file, target / file.name)
            copied += 1
    print(f"Subset disalin: {copied} file ke {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
