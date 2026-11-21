from research.analyze_detector_errors import match_detections


def test_match_detections_reports_class_confusion_at_same_location() -> None:
    ground_truth = [(0, (0.1, 0.1, 0.4, 0.4))]
    predictions = [(1, 0.91, (0.1, 0.1, 0.4, 0.4))]

    result = match_detections(ground_truth, predictions, iou_threshold=0.5)

    assert result["class_confusion"] == [{"actual": 0, "predicted": 1, "iou": 1.0}]
    assert result["false_negatives"] == []
    assert result["false_positives"] == []
