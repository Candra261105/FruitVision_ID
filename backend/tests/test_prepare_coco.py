import json
from pathlib import Path

from research.prepare_coco import convert_coco_source


def test_convert_coco_source_writes_only_target_classes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image = source / "data" / "test" / "IMG_001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    (source / "annotations.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "data/test/IMG_001.jpg", "width": 100, "height": 50}],
                "categories": [{"id": 1, "name": "tomato"}, {"id": 2, "name": "apple"}],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10]},
                    {"image_id": 1, "category_id": 2, "bbox": [1, 1, 5, 5]},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = convert_coco_source(source, source / "annotations.json", tmp_path / "out", ["tomato"])

    assert result == {"tomato": 1}
    assert (tmp_path / "out" / "images" / "IMG_001.jpg").is_file()
    fields = (tmp_path / "out" / "labels" / "IMG_001.txt").read_text(encoding="utf-8").split()
    assert fields[0] == "0"
    assert [float(value) for value in fields[1:]] == [0.2, 0.2, 0.2, 0.2]
    assert (tmp_path / "out" / "data.yaml").is_file()


def test_convert_coco_source_keeps_images_without_target_as_background(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image = source / "IMG_001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    annotations = source / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "IMG_001.jpg", "width": 100, "height": 50}],
                "categories": [{"id": 2, "name": "apple"}],
                "annotations": [{"image_id": 1, "category_id": 2, "bbox": [1, 1, 5, 5]}],
            }
        ),
        encoding="utf-8",
    )

    result = convert_coco_source(source, annotations, tmp_path / "out", ["tomato"])

    assert result == {}
    assert (tmp_path / "out" / "images" / "IMG_001.jpg").is_file()
    assert (tmp_path / "out" / "labels" / "IMG_001.txt").read_text(encoding="utf-8") == ""
