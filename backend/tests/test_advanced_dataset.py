import json
from pathlib import Path

import pytest
from PIL import Image

from research.advanced_manifest_schema import split_manifest, validate_manifest


def _write_image(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (120, 90, 60)).save(path)
    return path


def _write_taxonomy(path: Path) -> None:
    path.write_text(
        "tasks:\n  maturity: [unripe, ripe, overripe]\n",
        encoding="utf-8",
    )


def _record(image: str, session: str = "s1") -> dict[str, object]:
    return {
        "image": image,
        "object_id": f"{session}-obj-1",
        "fruit": "tomato",
        "bbox": [0.1, 0.1, 0.8, 0.8],
        "label": "ripe",
        "task": "maturity",
        "source": "local",
        "session_id": session,
        "annotator": "operator-1",
        "label_version": "v1",
        "license_or_permission": "internal",
    }


def test_validate_manifest_accepts_record_with_existing_image_and_taxonomy(tmp_path: Path) -> None:
    _write_image(tmp_path, "img.jpg")
    taxonomy = tmp_path / "taxonomy.yaml"
    _write_taxonomy(taxonomy)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_record("img.jpg")) + "\n", encoding="utf-8")

    report = validate_manifest(manifest, tmp_path, taxonomy)

    assert report.valid is True
    assert report.counts["maturity:ripe"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"image": "missing.jpg"},
        {"bbox": [0, 0, 1.2, 1]},
        {"task": "disease"},
        {"license_or_permission": ""},
    ],
)
def test_validate_manifest_rejects_invalid_records(tmp_path: Path, change: dict[str, object]) -> None:
    _write_image(tmp_path, "img.jpg")
    taxonomy = tmp_path / "taxonomy.yaml"
    _write_taxonomy(taxonomy)
    record = _record("img.jpg")
    record.update(change)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = validate_manifest(manifest, tmp_path, taxonomy)

    assert report.valid is False
    assert report.errors


def test_split_manifest_keeps_sessions_together(tmp_path: Path) -> None:
    records = [_record(f"{index}.jpg", session=f"session-{index}") for index in range(12)]

    split = split_manifest(records, seed=7)

    by_session = {}
    for record in split:
        by_session.setdefault(record["session_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_session.values())
    assert {record["split"] for record in split} == {"train", "validation", "test"}


def test_split_manifest_respects_pinned_splits() -> None:
    records = [
        {**_record("a.jpg", session="s-a"), "split": "train"},
        {**_record("b.jpg", session="s-b"), "split": "validation"},
        {**_record("c.jpg", session="s-c"), "split": "test"},
    ]

    split = split_manifest(records, seed=7)

    assert [record["split"] for record in split] == ["train", "validation", "test"]


def test_split_manifest_rejects_pinned_session_spanning_splits() -> None:
    records = [
        {**_record("a.jpg", session="same"), "split": "train"},
        {**_record("b.jpg", session="same"), "split": "test"},
    ]

    with pytest.raises(ValueError, match="lebih dari satu split"):
        split_manifest(records, seed=7)
