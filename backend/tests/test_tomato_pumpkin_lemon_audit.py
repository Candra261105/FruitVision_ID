from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.tomato_pumpkin_lemon_audit import (
    CLASS_NAMES,
    audit_sources,
    validate_yolo_label,
)


def test_class_order_is_the_public_contract() -> None:
    assert CLASS_NAMES == ("tomato", "pumpkin", "lemon")


def test_invalid_yolo_label_is_reported() -> None:
    assert validate_yolo_label("0 0.5 0.5 0.2 0.3", CLASS_NAMES) == []
    errors = validate_yolo_label("3 0.5 0.5 0 1.2", CLASS_NAMES)
    assert "class_id_out_of_range" in errors
    assert "box_out_of_range" in errors
    assert "box_zero_area" in errors


def test_audit_reports_partial_classes_without_inventing_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    (source / "images" / "one.jpg").write_bytes(b"not a real image, but audit is file based")
    (source / "labels" / "one.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    output = tmp_path / "audit.json"
    report = audit_sources([source], output, class_maps={"source": {0: "tomato"}})

    assert report["class_status"]["tomato"]["valid_box_count"] == 1
    assert report["class_status"]["pumpkin"]["valid_box_count"] == 0
    assert report["class_status"]["lemon"]["valid_box_count"] == 0
    assert report["partial"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["partial"] is True


def test_audit_rejects_missing_label(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "one.jpg").write_bytes(b"x")

    report = audit_sources([source], tmp_path / "audit.json")

    assert report["invalid_record_count"] == 1
    assert report["records"][0]["errors"] == ["missing_label"]
