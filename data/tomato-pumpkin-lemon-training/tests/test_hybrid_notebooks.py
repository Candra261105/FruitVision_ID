import ast
import json
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "notebooks"))

from notebook_helpers import find_package_root


def notebook_code(name: str) -> str:
    payload = json.loads((PACKAGE / "notebooks" / name).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"] if cell["cell_type"] == "code")


class HybridNotebookTests(unittest.TestCase):
    def test_find_package_root_from_nested_directory(self):
        self.assertEqual(find_package_root(PACKAGE / "dataset" / "images"), PACKAGE)

    def test_find_package_root_from_parent_directory(self):
        self.assertEqual(find_package_root(PACKAGE.parents[1]), PACKAGE)

    def test_inspection_notebook_contains_dataset_validation_and_visualization(self):
        code = notebook_code("01_dataset_inspection.ipynb")
        ast.parse(code)
        self.assertIn("dataset/data.yaml", code)
        self.assertIn("validate_dataset", code)
        self.assertIn("matplotlib", code)
        self.assertIn("tomato", code)
        self.assertIn("pumpkin", code)
        self.assertIn("lemon", code)

    def test_training_notebook_has_rtx2050_defaults_and_cuda_guard(self):
        code = notebook_code("02_train_tomato_pumpkin_lemon.ipynb")
        ast.parse(code)
        self.assertIn("train_auto.py", code)
        self.assertIn("EPOCHS = None", code)
        self.assertIn("BATCH = None", code)
        self.assertIn("RESUME = None", code)
        self.assertIn("requirements-training.txt", code)
        self.assertNotIn("external_holdout", code)

    def test_evaluation_notebook_separates_test_and_external_holdout(self):
        code = notebook_code("03_evaluate_model.ipynb")
        ast.parse(code)
        self.assertIn("best.pt", code)
        self.assertIn("data.yaml", code)
        self.assertIn("data-external.yaml", code)
        self.assertIn("split_manifest.json", code)
        self.assertIn("evaluate_tomato_pumpkin_lemon.py", code)
        self.assertIn("belum ada", code)


if __name__ == "__main__":
    unittest.main()
