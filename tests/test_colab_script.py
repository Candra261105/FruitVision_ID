from __future__ import annotations

import ast
from pathlib import Path
import zipfile
import pytest

from research.package_expanded_colab import package_dataset_for_colab


def test_colab_script_syntax_and_hyperparameters():
    script_path = Path("data/tomato-pumpkin-lemon-expanded/training/colab_train_expanded.py")
    assert script_path.exists()

    code = script_path.read_text(encoding="utf-8")
    # Verify valid Python syntax
    tree = ast.parse(code)
    assert tree is not None

    # Check key hyperparameters in code
    assert "imgsz=800" in code or "imgsz = 800" in code
    assert "batch=32" in code or "batch = 32" in code or "batch_size = 32" in code
    assert "yolo11x" in code or "yolo11s" in code
    assert "device=0" in code or "device = 0" in code
    assert "patience" in code and ("40" in code or "50" in code)
    assert "files.download" in code


def test_package_dataset_for_colab(tmp_path: Path):
    # Setup mock dataset structure
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "images" / "test").mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("names: [tomato, pumpkin, lemon]\n")

    training_dir = tmp_path / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "colab_train_expanded.py").write_text("print('test')\n")

    out_zip = tmp_path / "test_expanded.zip"
    result = package_dataset_for_colab(
        dataset_root=dataset_dir,
        training_script=training_dir / "colab_train_expanded.py",
        output_zip=out_zip,
    )

    assert result.exists()
    assert zipfile.is_zipfile(result)

    with zipfile.ZipFile(result, "r") as zf:
        namelist = zf.namelist()
        assert any("dataset/data.yaml" in name for name in namelist)
        assert any("colab_train_expanded.py" in name for name in namelist)
