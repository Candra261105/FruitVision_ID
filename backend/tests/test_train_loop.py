import json
from pathlib import Path

from research.train_loop import (
    augmentation_kwargs,
    find_resume_checkpoint,
    load_matrix,
    should_stop,
    summarize_metrics,
    write_json_atomic,
)
from research.detector_training import find_best_weight
from research.fruitvision_loop import detector_dataset_path, detector_gate


def test_stop_file_is_checked_between_experiments(tmp_path: Path) -> None:
    stop = tmp_path / "STOP_TRAINING"
    assert should_stop(stop) is False
    stop.write_text("stop", encoding="utf-8")
    assert should_stop(stop) is True


def test_matrix_rejects_duplicate_experiment_ids(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("experiments:\n  - id: same\n  - id: same\n", encoding="utf-8")
    try:
        load_matrix(matrix)
    except ValueError as exc:
        assert "unik" in str(exc)
    else:
        raise AssertionError("duplicate id harus ditolak")


def test_summarize_metrics_serializes_overall_and_per_class_values() -> None:
    class Box:
        map50 = 0.4
        map = 0.2
        mp = 0.5
        mr = 0.6
        p = [0.4, 0.6]
        r = [0.5, 0.7]
        ap50 = [0.3, 0.5]
        ap = [0.1, 0.3]

    class Metrics:
        box = Box()

    result = summarize_metrics(Metrics(), {0: "tomato", 1: "cucumber"})

    assert result["map50"] == 0.4
    assert result["map50_95"] == 0.2
    assert result["per_class"]["tomato"]["recall"] == 0.5
    assert result["per_class"]["cucumber"]["map50_95"] == 0.3


def test_augmentation_kwargs_selects_declared_policy_group() -> None:
    policy = {"groups": {"photo": {"hsv_h": 0.01, "hsv_s": 0.2, "hsv_v": 0.3}}}

    assert augmentation_kwargs({"augmentation": "photo"}, policy) == policy["groups"]["photo"]


def test_find_resume_checkpoint_finds_last_checkpoint_inside_run(tmp_path: Path) -> None:
    import torch

    weights = tmp_path / "weights"
    weights.mkdir()
    checkpoint = weights / "last.pt"
    torch.save({"epoch": 1}, checkpoint)

    assert find_resume_checkpoint(tmp_path) == checkpoint


def test_find_resume_checkpoint_ignores_corrupt_checkpoint(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "last.pt").write_bytes(b"not a pytorch checkpoint")

    assert find_resume_checkpoint(tmp_path) is None


def test_write_json_atomic_leaves_complete_json_file(tmp_path: Path) -> None:
    target = tmp_path / "status.json"

    write_json_atomic(target, {"status": "running", "step": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "running", "step": 2}


def test_corrupt_status_file_does_not_block_checkpoint_discovery(tmp_path: Path) -> None:
    from research.train_loop import load_existing_status

    status = tmp_path / "status.json"
    status.write_text("{broken", encoding="utf-8")

    assert load_existing_status(status) == {}


def test_advanced_runner_rejects_matrix_without_local_dataset_before_loading_model(tmp_path: Path) -> None:
    from research.advanced_train_loop import run_advanced_task

    manifest = tmp_path / "manifest.jsonl"
    image = tmp_path / "fruit.jpg"
    image.write_bytes(b"image")
    manifest.write_text(
        json.dumps(
            {
                "image": image.name,
                "object_id": "object-1",
                "fruit": "tomato",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "label": "ripe",
                "task": "maturity",
                "source": "source-a",
                "session_id": "session-a",
                "annotator": "annotator-a",
                "label_version": "v1",
                "license_or_permission": "CC BY 4.0",
            }
        ),
        encoding="utf-8",
    )

    result = run_advanced_task(
        {"id": "maturity", "task": "maturity", "model": "missing.pt"},
        manifest,
        tmp_path / "runs",
        "cpu",
    )

    assert result["status"] == "blocked_no_dataset"
    assert "data" in result["reason"]


def test_find_best_weight_resolves_ultralytics_detect_output(tmp_path: Path) -> None:
    expected = tmp_path / "runs" / "detect" / "runs" / "loop-detector" / "experiment-512" / "weights" / "best.pt"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"weights")

    result = find_best_weight(tmp_path / "runs" / "loop-detector", "experiment-512", tmp_path)

    assert result == expected.resolve()


def test_find_best_weight_raises_when_no_best_checkpoint_exists(tmp_path: Path) -> None:
    try:
        find_best_weight(tmp_path / "runs" / "loop-detector", "missing", tmp_path)
    except FileNotFoundError as exc:
        assert "best.pt" in str(exc)
    else:
        raise AssertionError("best.pt yang tidak ada harus menghasilkan error")


def test_detector_gate_requires_the_three_target_classes() -> None:
    assert detector_gate({"map50": 0.80, "per_class": {"tomato": 0.8, "cucumber": 0.8}}) is False
    assert detector_gate({"map50": 0.80, "per_class": {"tomato": 0.8, "cucumber": 0.8, "green beans": 0.39}}) is False
    assert detector_gate({"map50": 0.55, "per_class": {"tomato": 0.4, "cucumber": 0.4, "green beans": 0.4}}) is True


def test_detector_loop_uses_clean_three_class_experiment_dataset() -> None:
    assert detector_dataset_path().name == "combined-top3-cc0-experiment"
    assert detector_dataset_path().joinpath("data.yaml").is_file()


def test_yolo26m_dataset_validation_accepts_clean_three_class_dataset() -> None:
    from research.yolo26m_training import validate_detector_dataset

    result = validate_detector_dataset(detector_dataset_path() / "data.yaml")

    assert result["names"] == ["cucumber", "tomato", "green beans"]
    assert result["data_yaml"].endswith("combined-top3-cc0-experiment\\data.yaml")


def test_yolo26m_dataset_validation_rejects_wrong_class_names(tmp_path: Path) -> None:
    from research.yolo26m_training import validate_detector_dataset

    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [cucumber, tomato, beans]\n",
        encoding="utf-8",
    )

    try:
        validate_detector_dataset(dataset / "data.yaml")
    except ValueError as exc:
        assert "nama kelas" in str(exc)
    else:
        raise AssertionError("nama kelas yang salah harus ditolak")


def test_yolo26m_checkpoint_resolver_rejects_missing_best_checkpoint(tmp_path: Path) -> None:
    from research.yolo26m_training import resolve_yolo26m_checkpoint

    try:
        resolve_yolo26m_checkpoint(tmp_path, "missing-run")
    except FileNotFoundError as exc:
        assert "best.pt" in str(exc)
    else:
        raise AssertionError("best.pt yang belum ada harus ditolak")


def test_yolo26n_training_config_uses_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    from research.yolo26m_training import detector_training_config

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    result = detector_training_config()

    assert result == {"device": "cpu", "threads": 8}
