from research.targeted_epoch_runner import (
    PRESET_EPOCHS,
    build_run_name,
    validate_epoch_count,
)


def test_preset_epochs_include_50_to_100_by_tens() -> None:
    assert PRESET_EPOCHS == (50, 60, 70, 80, 90, 100)


def test_epoch_count_accepts_presets_and_future_values() -> None:
    assert validate_epoch_count(50) == 50
    assert validate_epoch_count(150) == 150


def test_epoch_count_rejects_zero_or_negative_values() -> None:
    for value in (0, -10):
        try:
            validate_epoch_count(value)
        except ValueError as exc:
            assert "positif" in str(exc)
        else:
            raise AssertionError("jumlah epoch tidak valid harus ditolak")


def test_run_name_contains_requested_epoch_count() -> None:
    assert build_run_name(80) == "continuation-512-yolo26n-from-epoch30-80ep"
