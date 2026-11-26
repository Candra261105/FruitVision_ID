"""Small helpers shared by the VS Code/Jupyter notebooks."""

from __future__ import annotations

from pathlib import Path


def find_package_root(start: Path | None = None) -> Path:
    """Find the training package by locating dataset/data.yaml."""
    current = (start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "dataset" / "data.yaml").is_file():
            return candidate
    for data_yaml in current.glob("**/dataset/data.yaml"):
        return data_yaml.parent.parent
    raise FileNotFoundError(
        "Folder paket tidak ditemukan. Buka folder "
        "tomato-pumpkin-lemon-training di VS Code."
    )
