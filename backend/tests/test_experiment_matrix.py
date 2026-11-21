from pathlib import Path

import yaml


def test_experiment_matrix_has_unique_ids_and_bounded_augmentation() -> None:
    matrix = yaml.safe_load(Path("research/experiment_matrix.yaml").read_text(encoding="utf-8"))
    experiments = matrix["experiments"]
    assert len({item["id"] for item in experiments}) == len(experiments)
    assert all(1 <= item["epochs"] <= 500 for item in experiments)
    assert all(0 <= item["patience"] < item["epochs"] for item in experiments)


def test_photometric_augmentation_does_not_use_extreme_hue_shift() -> None:
    policy = yaml.safe_load(Path("research/augmentation_policy.yaml").read_text(encoding="utf-8"))
    assert policy["groups"]["light-dark-contrast-color"]["hsv_h"] <= 0.03
