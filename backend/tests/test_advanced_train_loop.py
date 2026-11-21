import json
from pathlib import Path

from research.advanced_train_loop import (
    StopFileCallback,
    find_valid_checkpoint,
    load_advanced_matrix,
    run_advanced_task,
    write_status_atomic,
)


def test_missing_manifest_blocks_task_without_creating_weights(tmp_path: Path) -> None:
    result = run_advanced_task(
        {"id": "maturity", "task": "maturity"},
        tmp_path / "missing.jsonl",
        tmp_path / "runs",
        "cpu",
    )

    assert result["status"] == "blocked_no_dataset"
    assert not list((tmp_path / "runs").rglob("*.pt"))


def test_write_status_atomic_writes_parseable_json(tmp_path: Path) -> None:
    path = tmp_path / "status.json"

    write_status_atomic(path, {"status": "interrupted"})

    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "interrupted"


def test_corrupt_checkpoint_is_ignored(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "last.pt"
    checkpoint.write_bytes(b"invalid")

    assert find_valid_checkpoint(tmp_path) is None


def test_stop_file_callback_stops_only_when_file_exists(tmp_path: Path) -> None:
    stop_file = tmp_path / "STOP_TRAINING"
    callback = StopFileCallback(stop_file)

    class Trainer:
        stop = False

    trainer = Trainer()
    callback(trainer)
    assert trainer.stop is False
    assert callback.triggered is False

    stop_file.write_text("stop", encoding="utf-8")
    callback(trainer)
    assert trainer.stop is True
    assert callback.triggered is True


def test_matrix_has_four_independent_tasks(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "tasks:\n"
        "  - id: disease\n    task: disease\n"
        "  - id: maturity\n    task: maturity\n"
        "  - id: rot\n    task: rot\n"
        "  - id: grade\n    task: grade\n",
        encoding="utf-8",
    )

    result = load_advanced_matrix(matrix)

    assert [item["task"] for item in result] == ["disease", "maturity", "rot", "grade"]
