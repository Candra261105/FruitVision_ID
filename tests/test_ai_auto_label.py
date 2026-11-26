from __future__ import annotations

import pytest

from research.ai_auto_label import (
    calculate_box_iou,
    filter_and_deduplicate_boxes,
    generate_yolo_label_lines,
)


def test_calculate_box_iou():
    box1 = (0.5, 0.5, 0.2, 0.2)
    box2 = (0.5, 0.5, 0.2, 0.2)
    # Identical boxes should have IoU == 1.0
    assert calculate_box_iou(box1, box2) == pytest.approx(1.0)

    # Disjoint boxes
    box3 = (0.1, 0.1, 0.1, 0.1)
    assert calculate_box_iou(box1, box3) == pytest.approx(0.0)


def test_filter_and_deduplicate_boxes():
    # Format: (cls, cx, cy, w, h, conf)
    boxes = [
        (0, 0.5, 0.5, 0.2, 0.2, 0.90),
        (0, 0.501, 0.501, 0.199, 0.199, 0.85),  # duplicate of first box
        (2, 0.1, 0.1, 0.05, 0.05, 0.40),       # low conf (< 0.6)
        (1, 0.8, 0.8, 0.3, 0.3, 0.95),         # valid pumpkin
        (0, 1.2, 0.5, 0.2, 0.2, 0.90),         # out of bounds cx
    ]
    filtered = filter_and_deduplicate_boxes(boxes, min_conf=0.6, iou_thresh=0.5)
    assert len(filtered) == 2
    # Highest confidence candidate comes first
    assert filtered[0][0] == 1
    assert filtered[0][5] == pytest.approx(0.95)
    assert filtered[1][0] == 0
    assert filtered[1][5] == pytest.approx(0.90)


def test_generate_yolo_label_lines():
    boxes = [
        (0, 0.5, 0.5, 0.2, 0.2, 0.9),
        (2, 0.25, 0.75, 0.1, 0.15, 0.8),
    ]
    lines = generate_yolo_label_lines(boxes)
    expected_lines = [
        "0 0.500000 0.500000 0.200000 0.200000",
        "2 0.250000 0.750000 0.100000 0.150000",
    ]
    assert lines.strip() == "\n".join(expected_lines)


def test_ingest_kitchen_coco_dataset(tmp_path):
    from PIL import Image
    import json
    from research.harvest_web_context import ingest_kitchen_coco_dataset

    # Create dummy coco json and image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    dummy_img = img_dir / "IMG_001.jpg"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(dummy_img)

    coco_data = {
        "images": [{"id": 1, "file_name": "IMG_001.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20]}, # lemon -> 2
            {"id": 11, "image_id": 1, "category_id": 13, "bbox": [50, 50, 30, 30]}, # tomato -> 0
        ],
    }
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(coco_data))

    out_img = tmp_path / "out_img"
    out_lbl = tmp_path / "out_lbl"
    records = ingest_kitchen_coco_dataset(json_path, img_dir, out_img, out_lbl)
    assert len(records) == 1
    assert (out_img / "kitchen_IMG_001.jpg").is_file()
    assert (out_lbl / "kitchen_IMG_001.txt").is_file()
    lbl_content = (out_lbl / "kitchen_IMG_001.txt").read_text()
    assert "2 0.200000 0.200000 0.200000 0.200000" in lbl_content
    assert "0 0.650000 0.650000 0.300000 0.300000" in lbl_content
