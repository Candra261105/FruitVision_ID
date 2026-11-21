from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_detections(
    ground_truth: list[tuple[int, tuple[float, float, float, float]]],
    predictions: list[tuple[int, float, tuple[float, float, float, float]]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    pairs: list[tuple[float, int, int]] = []
    for gt_index, (_, gt_box) in enumerate(ground_truth):
        for prediction_index, (_, _, prediction_box) in enumerate(predictions):
            pairs.append((_iou(gt_box, prediction_box), gt_index, prediction_index))
    pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    confusion: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    for overlap, gt_index, prediction_index in pairs:
        if overlap < iou_threshold or gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        actual = ground_truth[gt_index][0]
        predicted = predictions[prediction_index][0]
        if actual == predicted:
            matched.append({"class": actual, "iou": round(overlap, 6)})
        else:
            confusion.append({"actual": actual, "predicted": predicted, "iou": round(overlap, 6)})
    false_negatives = [ground_truth[index][0] for index in range(len(ground_truth)) if index not in used_gt]
    false_positives = [predictions[index][0] for index in range(len(predictions)) if index not in used_predictions]
    return {
        "matched": matched,
        "class_confusion": confusion,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def _read_labels(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    result: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id, cx, cy, width, height = map(float, fields)
        result.append((int(class_id), (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)))
    return result


def analyze(weights: Path, dataset: Path, output: Path, imgsz: int, confidence: float) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(dataset.read_text(encoding="utf-8")) or {}
    root = Path(config.get("path", dataset.parent))
    if not root.is_absolute():
        root = (dataset.parent / root).resolve()
    split = config.get("test", "images/test")
    images_dir = root / split
    labels_dir = root / "labels" / Path(split).name
    model = YOLO(str(weights))
    names = {int(key): str(value) for key, value in model.names.items()} if isinstance(model.names, dict) else dict(enumerate(model.names))
    summary: dict[str, Any] = {"weights": str(weights), "dataset": str(dataset), "split": str(split), "images": 0, "ground_truth": 0, "predictions": 0, "per_class": {}, "errors": []}
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        ground_truth = _read_labels(labels_dir / f"{image_path.stem}.txt")
        result = model.predict(str(image_path), conf=confidence, imgsz=imgsz, device="cpu", verbose=False)[0]
        width, height = Image.open(image_path).size
        predictions: list[tuple[int, float, tuple[float, float, float, float]]] = []
        if result.boxes is not None:
            for box, score, class_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                predictions.append((int(class_id), float(score), (box[0] / width, box[1] / height, box[2] / width, box[3] / height)))
        matched = match_detections(ground_truth, predictions)
        summary["images"] += 1
        summary["ground_truth"] += len(ground_truth)
        summary["predictions"] += len(predictions)
        for class_id, _ in ground_truth:
            entry = summary["per_class"].setdefault(names.get(class_id, str(class_id)), {"ground_truth": 0, "false_negatives": 0, "false_positives": 0, "confusion": 0})
            entry["ground_truth"] += 1
        for class_id in matched["false_negatives"]:
            summary["per_class"].setdefault(names.get(class_id, str(class_id)), {"ground_truth": 0, "false_negatives": 0, "false_positives": 0, "confusion": 0})["false_negatives"] += 1
        for class_id in matched["false_positives"]:
            summary["per_class"].setdefault(names.get(class_id, str(class_id)), {"ground_truth": 0, "false_negatives": 0, "false_positives": 0, "confusion": 0})["false_positives"] += 1
        for item in matched["class_confusion"]:
            summary["per_class"].setdefault(names.get(item["actual"], str(item["actual"])), {"ground_truth": 0, "false_negatives": 0, "false_positives": 0, "confusion": 0})["confusion"] += 1
        if matched["false_negatives"] or matched["false_positives"] or matched["class_confusion"]:
            summary["errors"].append({"image": str(image_path), **matched})
    summary["error_images"] = len(summary["errors"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(analyze(args.weights, args.data, args.output, args.imgsz, args.conf), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
