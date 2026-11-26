"""AI auto-labeling, Non-Maximum Suppression (NMS) deduplication, and YOLO formatter."""

from __future__ import annotations

from typing import Sequence


def calculate_box_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Calculate Intersection over Union (IoU) of two (cx, cy, w, h) boxes."""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    x1_min, x1_max = cx1 - w1 / 2.0, cx1 + w1 / 2.0
    y1_min, y1_max = cy1 - h1 / 2.0, cy1 + h1 / 2.0

    x2_min, x2_max = cx2 - w2 / 2.0, cx2 + w2 / 2.0
    y2_min, y2_max = cy2 - h2 / 2.0, cy2 + h2 / 2.0

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)
    union_area = area1 + area2 - inter_area

    if union_area <= 1e-9:
        return 0.0
    return inter_area / union_area


def filter_and_deduplicate_boxes(
    boxes: Sequence[tuple[int, float, float, float, float, float]],
    min_conf: float = 0.6,
    iou_thresh: float = 0.5,
) -> list[tuple[int, float, float, float, float, float]]:
    """Filter low-confidence, invalid coordinates, and deduplicate overlapping boxes per class."""
    valid: list[tuple[int, float, float, float, float, float]] = []
    for cls_id, cx, cy, w, h, conf in boxes:
        if conf < min_conf:
            continue
        # Bounds check
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        valid.append((int(cls_id), float(cx), float(cy), float(w), float(h), float(conf)))

    # Sort descending by confidence
    valid.sort(key=lambda b: b[5], reverse=True)

    kept: list[tuple[int, float, float, float, float, float]] = []
    for candidate in valid:
        c_cls, c_cx, c_cy, c_w, c_h, _ = candidate
        c_box = (c_cx, c_cy, c_w, c_h)
        duplicate = False
        for existing in kept:
            e_cls, e_cx, e_cy, e_w, e_h, _ = existing
            if e_cls != c_cls:
                continue
            e_box = (e_cx, e_cy, e_w, e_h)
            if calculate_box_iou(c_box, e_box) > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return kept


def generate_yolo_label_lines(
    boxes: Sequence[tuple[int, float, float, float, float, ...]],
) -> str:
    """Convert detections into standard YOLO label format string."""
    lines: list[str] = []
    for item in boxes:
        cls_id = int(item[0])
        cx = min(max(float(item[1]), 0.0), 1.0)
        cy = min(max(float(item[2]), 0.0), 1.0)
        w = min(max(float(item[3]), 0.0), 1.0)
        h = min(max(float(item[4]), 0.0), 1.0)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + "\n" if lines else ""
