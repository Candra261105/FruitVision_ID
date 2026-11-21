from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.advanced_manifest_schema import validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi manifest dataset lanjutan FruitVision.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("research/advanced_taxonomy.yaml"))
    args = parser.parse_args()
    report = validate_manifest(args.manifest, args.images, args.taxonomy)
    print(json.dumps({"valid": report.valid, "errors": report.errors, "counts": report.counts}, ensure_ascii=False, indent=2))
    return 0 if report.valid else 1 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
