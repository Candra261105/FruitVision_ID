import json
from pathlib import Path

from research.gate_advanced_model import write_advanced_gate
from research.evaluate_advanced import summarize_classification


def test_summarize_classification_reports_macro_metrics_and_ordinal_error() -> None:
    result = summarize_classification(
        task="maturity",
        y_true=["unripe", "ripe", "overripe", "ripe"],
        y_pred=["unripe", "overripe", "overripe", "ripe"],
        labels=["unripe", "ripe", "overripe"],
    )

    assert 0 <= result["macro_f1"] <= 1
    assert result["ordinal_mean_error"] == 0.25
    assert result["support"] == 4
    assert result["labels"] == ["unripe", "ripe", "overripe"]


def test_gate_accepts_partial_label_subset_for_disease(tmp_path: Path) -> None:
    model = tmp_path / "disease.pt"
    model.write_bytes(b"model")
    import hashlib

    evaluation = {
        "task": "disease",
        "labels": ["healthy", "belly_rot", "pythium_fruit_rot"],
        "macro_f1": 0.86,
        "balanced_accuracy": 0.87,
        "support": 434,
        "per_class": {
            "healthy": {"support": 143},
            "belly_rot": {"support": 143},
            "pythium_fruit_rot": {"support": 148},
        },
    }
    provenance = {
        "task": "disease",
        "manifest_sha256": "a" * 64,
        "taxonomy_sha256": "b" * 64,
        "model_sha256": hashlib.sha256(b"model").hexdigest(),
        "taxonomy_version": 1,
        "split_strategy": "average-hash session_id",
    }

    payload = json.loads(write_advanced_gate(model, evaluation, provenance).read_text(encoding="utf-8"))

    assert payload["status"] == "eligible"
    assert payload["reasons"] == []


def test_gate_rejects_missing_provenance_and_low_metrics(tmp_path: Path) -> None:
    model = tmp_path / "maturity.pt"
    model.write_bytes(b"model")

    gate = write_advanced_gate(
        model,
        {"macro_f1": 0.4, "balanced_accuracy": 0.4, "support": 20},
        {},
    )

    payload = json.loads(gate.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["reasons"]


def test_gate_does_not_mark_small_dataset_eligible(tmp_path: Path) -> None:
    model = tmp_path / "grade.pt"
    model.write_bytes(b"model")

    gate = write_advanced_gate(
        model,
        {"task": "grade", "macro_f1": 0.95, "balanced_accuracy": 0.95, "support": 3},
        {
            "manifest_sha256": "a" * 64,
            "taxonomy_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "taxonomy_version": 1,
            "task": "grade",
        },
    )

    assert json.loads(gate.read_text(encoding="utf-8"))["status"] == "rejected"


def test_gate_rejects_task_mismatch_and_invalid_hashes(tmp_path: Path) -> None:
    model = tmp_path / "grade.pt"
    model.write_bytes(b"model")

    gate = write_advanced_gate(
        model,
        {"task": "maturity", "macro_f1": 0.95, "balanced_accuracy": 0.95, "support": 20},
        {
            "manifest_sha256": "not-a-sha256",
            "taxonomy_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "taxonomy_version": 1,
            "task": "grade",
        },
    )

    payload = json.loads(gate.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert any("task" in reason or "sha256" in reason for reason in payload["reasons"])


def test_gate_rejects_non_finite_metrics_without_raising(tmp_path: Path) -> None:
    model = tmp_path / "rot.pt"
    model.write_bytes(b"model")

    gate = write_advanced_gate(
        model,
        {
            "task": "rot",
            "macro_f1": float("nan"),
            "balanced_accuracy": 0.9,
            "support": 20,
            "labels": ["not_rotten", "rotten"],
            "per_class": {"not_rotten": {"support": 10}, "rotten": {"support": 10}},
        },
        {
            "manifest_sha256": "a" * 64,
            "taxonomy_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "taxonomy_version": 1,
            "task": "rot",
            "split_strategy": "source+session_id",
        },
    )

    assert json.loads(gate.read_text(encoding="utf-8"))["status"] == "rejected"
