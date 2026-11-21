from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_images(dataset: Path) -> dict[str, tuple[str, str]]:
    indexed: dict[str, tuple[str, str]] = {}
    image_root = dataset / "images"
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        relative = path.relative_to(image_root)
        split = relative.parts[0] if len(relative.parts) > 1 else "unspecified"
        indexed[_image_hash(path)] = (split, str(path))
    return indexed


def audit_cross_source_duplicates(old_dataset: Path, new_dataset: Path) -> dict[str, Any]:
    old = _index_images(old_dataset)
    new = _index_images(new_dataset)
    shared = sorted(set(old) & set(new))
    split_collisions = Counter(f"{old[h][0]}->{new[h][0]}" for h in shared)
    return {
        "old_image_count": len(old),
        "new_image_count": len(new),
        "old_unique_hashes": len(old),
        "new_unique_hashes": len(new),
        "cross_exact_duplicates": len(shared),
        "split_collisions": dict(sorted(split_collisions.items())),
        "examples": [
            {"old": old[h][1], "new": new[h][1], "old_split": old[h][0], "new_split": new[h][0]}
            for h in shared[:10]
        ],
    }
