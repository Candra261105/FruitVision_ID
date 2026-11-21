from pathlib import Path

from research.prepare_top3 import load_source_names, remap_yolo_text


def test_remap_yolo_text_keeps_only_top3_and_maps_local_names() -> None:
    result = remap_yolo_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
        ["Tomato", "cucumber"],
        ["cucumber", "tomato", "green beans"],
    )

    assert result == "1 0.5 0.5 0.2 0.2\n0 0.4 0.4 0.1 0.1\n"


def test_find_parallel_label_supports_images_and_labels_layout(tmp_path: Path) -> None:
    from research.prepare_top3 import find_parallel_label

    image = tmp_path / "images" / "test" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    label = tmp_path / "labels" / "test" / "a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    assert find_parallel_label(tmp_path, image) == label


def test_load_source_names_accepts_yolo_yaml(tmp_path: Path) -> None:
    config = tmp_path / "data.yaml"
    config.write_text("names:\n  0: Tomato\n  1: cucumber\n", encoding="utf-8")

    assert load_source_names(config) == ["Tomato", "cucumber"]
