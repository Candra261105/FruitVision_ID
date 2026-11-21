import json
from pathlib import Path

from research.select_model import select_model


def _write_run(root: Path, name: str, metrics: dict, kitchen: dict | None = None) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if kitchen is not None:
        (run / "kitchen-metrics.json").write_text(json.dumps(kitchen), encoding="utf-8")


def test_select_model_rejects_high_map_with_low_class_recall(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "high-map-low-recall",
        {
            "map50": 0.8,
            "map50_95": 0.6,
            "per_class": {
                "cucumber": {"recall": 0.1},
                "tomato": {"recall": 0.8},
                "green beans": {"recall": 0.8},
            },
        },
        {"map50": 0.5},
    )

    result = select_model(tmp_path, required_classes=["cucumber", "tomato", "green beans"], min_class_recall=0.3)

    candidate = result["candidates"][0]
    assert candidate["status"] == "rejected"
    assert "cucumber" in candidate["reasons"][0]


def test_select_model_requires_kitchen_metrics(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "candidate-without-kitchen",
        {
            "map50": 0.8,
            "map50_95": 0.6,
            "per_class": {
                "cucumber": {"recall": 0.8},
                "tomato": {"recall": 0.8},
                "green beans": {"recall": 0.8},
            },
        },
    )

    result = select_model(tmp_path, required_classes=["cucumber", "tomato", "green beans"])

    candidate = result["candidates"][0]
    assert candidate["status"] == "candidate"
    assert "Kitchen" in candidate["reasons"][0]
