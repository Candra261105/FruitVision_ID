from __future__ import annotations

from collections import Counter
from typing import Any


def summarize_classification(
    task: str,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("panjang y_true dan y_pred berbeda")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for actual, predicted in zip(y_true, y_pred):
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(y_true, y_pred))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(y_true, y_pred))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1_values.append(f1)
        recalls.append(recall)
    ordinal_mean_error = 0.0
    if task in {"maturity", "grade"} and y_true:
        positions = {label: index for index, label in enumerate(labels)}
        ordinal_mean_error = sum(abs(positions.get(actual, -1) - positions.get(predicted, -1)) for actual, predicted in zip(y_true, y_pred)) / len(y_true)
    return {
        "task": task,
        "labels": list(labels),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "ordinal_mean_error": ordinal_mean_error,
        "support": len(y_true),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
