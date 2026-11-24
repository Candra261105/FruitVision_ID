from __future__ import annotations

from pathlib import Path

import pytest

from research.evaluate_tomato_pumpkin_lemon import build_gate, summarize_metrics


def test_gate_rejects_missing_required_class() -> None:
    gate = build_gate(
        weights=Path("best.pt"),
        dataset_manifest=Path("manifest.json"),
        metrics={
            "map50": 0.8,
            "map50_95": 0.5,
            "per_class": {"tomato": {"support": 10}, "pumpkin": {"support": 10}},
        },
    )

    assert gate["status"] == "partial"
    assert "lemon" in gate["reasons"][0]


def test_gate_rejects_weak_class_even_when_overall_metric_is_high() -> None:
    gate = build_gate(
        weights=Path("best.pt"),
        dataset_manifest=Path("manifest.json"),
        metrics={
            "map50": 0.9,
            "map50_95": 0.7,
            "per_class": {
                "tomato": {"support": 20, "map50": 0.8},
                "pumpkin": {"support": 20, "map50": 0.8},
                "lemon": {"support": 20, "map50": 0.2},
            },
        },
    )

    assert gate["status"] == "rejected"
    assert any("lemon" in reason for reason in gate["reasons"])


def test_metric_summary_keeps_all_three_classes() -> None:
    class Box:
        map50 = 0.5
        map = 0.3
        mp = 0.6
        mr = 0.7
        p = [0.8, 0.6, 0.4]
        r = [0.7, 0.5, 0.3]
        ap50 = [0.75, 0.55, 0.35]
        ap = [0.5, 0.3, 0.1]
        ap_class_index = [0, 1, 2]

    class Metrics:
        box = Box()

    result = summarize_metrics(Metrics(), ["tomato", "pumpkin", "lemon"])

    assert list(result["per_class"]) == ["tomato", "pumpkin", "lemon"]
    assert result["per_class"]["lemon"]["map50_95"] == 0.1
