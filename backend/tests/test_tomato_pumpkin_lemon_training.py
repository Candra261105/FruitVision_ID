from __future__ import annotations

from pathlib import Path

import pytest

from research.train_tomato_pumpkin_lemon import validate_training_request


def test_training_request_requires_exact_class_order(tmp_path: Path) -> None:
    data = tmp_path / "data.yaml"
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "images" / "test").mkdir(parents=True)
    data.write_text("names: [tomato, pumpkin, lemon]\ntrain: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
    assert validate_training_request(data, 50)["epochs"] == 50


def test_training_request_rejects_invalid_epoch(tmp_path: Path) -> None:
    data = tmp_path / "data.yaml"
    data.write_text("names: [tomato, pumpkin, lemon]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="epochs"):
        validate_training_request(data, 0)


def test_training_request_rejects_wrong_taxonomy(tmp_path: Path) -> None:
    data = tmp_path / "data.yaml"
    data.write_text("names: [cucumber, tomato, green beans]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="class order"):
        validate_training_request(data, 100)
