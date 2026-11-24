"""Compatibility entry point for the new tomato/pumpkin/lemon trainer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.train_tomato_pumpkin_lemon import main


if __name__ == "__main__":
    raise SystemExit(main())
