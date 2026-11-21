from research.evaluate import build_evaluation_result
from research.fruitvision_loop import detector_gate


def test_build_evaluation_result_contains_per_class_metrics() -> None:
    class Box:
        map50 = 0.4
        map = 0.2
        mp = 0.5
        mr = 0.6
        ap_class_index = [0, 1]
        p = [0.4, 0.6]
        r = [0.5, 0.7]
        ap50 = [0.3, 0.5]
        ap = [0.1, 0.3]

    class Metrics:
        box = Box()

    result = build_evaluation_result(Metrics(), {0: "tomato", 1: "cucumber"}, "model.pt", "data.yaml", "cpu", 1.2)

    assert result["map50"] == 0.4
    assert result["per_class"]["cucumber"]["recall"] == 0.7


def test_detector_gate_rejects_incomplete_external_holdout_metrics() -> None:
    metrics = {
        "map50": 0.81,
        "per_class": {"cucumber": 0.81},
        "split": "external_holdout",
    }

    assert detector_gate(metrics) is False
