from pathlib import Path

import pytest

from research.download_sources import validate_source_entry


def test_validate_source_entry_rejects_unknown_license() -> None:
    with pytest.raises(ValueError, match="lisensi"):
        validate_source_entry({"id": "x", "license": "unknown", "annotation": "object-detection"})


def test_validate_source_entry_rejects_classification_only() -> None:
    with pytest.raises(ValueError, match="classification"):
        validate_source_entry({"id": "x", "license": "MIT", "annotation": "image-folder"})


def test_validate_source_entry_accepts_object_detection(tmp_path: Path) -> None:
    entry = {"id": "x", "license": "MIT", "annotation": "yolo-bounding-box", "url": "https://example.com/x.zip"}
    assert validate_source_entry(entry)["id"] == "x"
