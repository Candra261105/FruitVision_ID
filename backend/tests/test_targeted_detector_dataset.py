import hashlib
from pathlib import Path

from research.build_targeted_detector_dataset import build_targeted_dataset


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_targeted_dataset_keeps_non_train_splits_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val", "test", "external_holdout"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        (source / "images" / split / "one.jpg").write_bytes(f"image-{split}".encode())
        label = "2 0.5 0.5 0.2 0.2" if split == "train" else "0 0.5 0.5 0.2 0.2"
        (source / "labels" / split / "one.txt").write_text(label, encoding="utf-8")
    (source / "data.yaml").write_text("path: .\nnames: [cucumber, tomato, green beans]\n", encoding="utf-8")
    output = tmp_path / "output"

    result = build_targeted_dataset(source, output, tmp_path / "report.json", target_extra_fraction=1.0)

    assert result["duplicated_green_beans_images"] == 1
    assert all(result["unchanged_splits"].values())
    for split in ("val", "test", "external_holdout"):
        assert _hash(source / "images" / split / "one.jpg") == _hash(output / "images" / split / "one.jpg")
        assert _hash(source / "labels" / split / "one.txt") == _hash(output / "labels" / split / "one.txt")
