import json
from pathlib import Path

from research.build_combined_dataset import build_data_yaml, choose_unique_records, remap_label_lines, split_hashes


def test_split_hashes_never_places_same_hash_in_two_splits() -> None:
    result = split_hashes(["a", "b", "c", "d"], seed=20261118)
    assert set(result["train"]).isdisjoint(result["val"])
    assert set(result["train"]).isdisjoint(result["test"])
    assert set(result["val"]).isdisjoint(result["test"])


def test_build_data_yaml_contains_only_top3(tmp_path: Path) -> None:
    output = tmp_path / "data.yaml"
    build_data_yaml(output, ["tomato", "potato", "banana"])
    payload = output.read_text(encoding="utf-8")
    assert "tomato" in payload
    assert "potato" in payload
    assert "banana" in payload


def test_remap_label_lines_uses_source_names_and_drops_non_top3() -> None:
    source_names = ["cucumber", "tomato", "green beans"]
    top3 = ["tomato", "cucumber", "green beans"]

    result = remap_label_lines(
        "0 0.1 0.2 0.3 0.4\n1 0.5 0.5 0.2 0.2\n",
        source_names,
        top3,
    )

    assert result == "1 0.1 0.2 0.3 0.4\n0 0.5 0.5 0.2 0.2\n"


def test_choose_unique_records_keeps_one_record_per_image_hash() -> None:
    records = [
        {"hash": "same", "source": "small", "label_text": "0 0.5 0.5 0.1 0.1\n"},
        {"hash": "same", "source": "large", "label_text": "0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n"},
        {"hash": "other", "source": "other", "label_text": "1 0.5 0.5 0.1 0.1\n"},
    ]

    result = choose_unique_records(records)

    assert {item["source"] for item in result} == {"large", "other"}
    assert len(result) == 2
