import json
from pathlib import Path

from app.services.advanced_model_registry import load_advanced_models


def test_missing_or_rejected_gates_remain_unavailable(tmp_path: Path) -> None:
    (tmp_path / "maturity.pt").write_bytes(b"model")
    (tmp_path / "maturity.gate.json").write_text(
        json.dumps({"status": "rejected", "provenance": {"task": "maturity"}}),
        encoding="utf-8",
    )

    models = load_advanced_models(tmp_path)

    assert models["maturity"].status == "rejected"
    assert models["disease"].status == "unavailable"


def test_eligible_model_is_loaded_only_for_matching_task(tmp_path: Path) -> None:
    taxonomy = tmp_path / "research" / "advanced_taxonomy.yaml"
    taxonomy.parent.mkdir()
    taxonomy.write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "maturity.pt").write_bytes(b"model")
    import hashlib

    model_sha256 = hashlib.sha256(b"model").hexdigest()
    taxonomy_sha256 = hashlib.sha256(taxonomy.read_bytes()).hexdigest()
    (tmp_path / "maturity.gate.json").write_text(
        json.dumps(
            {
                "status": "eligible",
                    "evaluation": {
                        "task": "maturity",
                        "labels": ["unripe", "half_ripe", "ripe"],
                        "macro_f1": 0.9,
                        "balanced_accuracy": 0.9,
                        "support": 60,
                        "per_class": {
                            "unripe": {"support": 20},
                            "half_ripe": {"support": 20},
                            "ripe": {"support": 20},
                        },
                    },
                "provenance": {
                    "task": "maturity",
                    "taxonomy_version": 1,
                    "manifest_sha256": "a" * 64,
                    "taxonomy_sha256": taxonomy_sha256,
                    "model_sha256": model_sha256,
                    "split_strategy": "source+session_id",
                    "fruits": ["tomato"],
                    "imgsz": 160,
                },
            }
        ),
        encoding="utf-8",
    )

    models = load_advanced_models(tmp_path)

    assert models["maturity"].status == "eligible"
    assert models["maturity"].runnable is True
    assert models["maturity"].fruits == ("tomato",)
    assert "aktif untuk tomato" in models["maturity"].detail


def test_disease_gate_accepts_subset_of_taxonomy_labels(tmp_path: Path) -> None:
    taxonomy = tmp_path / "research" / "advanced_taxonomy.yaml"
    taxonomy.parent.mkdir()
    taxonomy.write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "disease.pt").write_bytes(b"model")
    import hashlib

    model_sha256 = hashlib.sha256(b"model").hexdigest()
    taxonomy_sha256 = hashlib.sha256(taxonomy.read_bytes()).hexdigest()
    (tmp_path / "disease.gate.json").write_text(
        json.dumps(
            {
                "status": "eligible",
                "evaluation": {
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
                },
                "provenance": {
                    "task": "disease",
                    "taxonomy_version": 1,
                    "manifest_sha256": "a" * 64,
                    "taxonomy_sha256": taxonomy_sha256,
                    "model_sha256": model_sha256,
                    "split_strategy": "average-hash session_id",
                    "fruits": ["cucumber"],
                    "imgsz": 160,
                },
            }
        ),
        encoding="utf-8",
    )

    models = load_advanced_models(tmp_path)

    assert models["disease"].status == "eligible"
    assert models["disease"].labels == ("healthy", "belly_rot", "pythium_fruit_rot")
    assert "cucumber" in models["disease"].detail


def test_eligible_gate_is_rejected_when_taxonomy_hash_does_not_match(tmp_path: Path) -> None:
    taxonomy = tmp_path / "research" / "advanced_taxonomy.yaml"
    taxonomy.parent.mkdir()
    taxonomy.write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "rot.pt").write_bytes(b"model")
    model_sha256 = __import__("hashlib").sha256(b"model").hexdigest()
    (tmp_path / "rot.gate.json").write_text(
        json.dumps(
            {
                "status": "eligible",
                "evaluation": {
                    "task": "rot",
                    "labels": ["not_rotten", "rotten"],
                    "macro_f1": 0.9,
                    "balanced_accuracy": 0.9,
                    "support": 40,
                    "per_class": {"not_rotten": {"support": 20}, "rotten": {"support": 20}},
                },
                "provenance": {
                    "task": "rot",
                    "taxonomy_version": 1,
                    "manifest_sha256": "a" * 64,
                    "taxonomy_sha256": "b" * 64,
                    "model_sha256": model_sha256,
                    "split_strategy": "source+session_id",
                },
            }
        ),
        encoding="utf-8",
    )

    models = load_advanced_models(tmp_path)

    assert models["rot"].status == "rejected"
    assert models["disease"].status == "unavailable"


def test_malformed_json_shapes_do_not_break_registry_startup(tmp_path: Path) -> None:
    for task, payload in (("disease", "null"), ("maturity", "[]"), ("rot", '{"provenance": []}')):
        (tmp_path / f"{task}.pt").write_bytes(b"model")
        (tmp_path / f"{task}.gate.json").write_text(payload, encoding="utf-8")

    models = load_advanced_models(tmp_path)

    assert all(models[task].status == "rejected" for task in ("disease", "maturity", "rot"))
