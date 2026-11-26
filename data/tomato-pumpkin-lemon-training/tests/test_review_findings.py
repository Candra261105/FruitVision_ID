import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "training"))

from dataset_validation import validate_dataset
from evaluate_tomato_pumpkin_lemon import validate_evaluation_inputs


class ReviewFindingsTests(unittest.TestCase):
    def test_validation_reports_synthetic_manifest_and_orphan_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "images" / "train" / "one.jpg").write_bytes(b"image")
            (root / "labels" / "train" / "one.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            for split in ("val", "test", "external_holdout"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            (root / "synthetic_manifest.json").write_text(
                json.dumps({"items": [{"output_image": "images/train/one.jpg", "output_label": "labels/train/one.txt", "split": "train", "synthetic": True}]}),
                encoding="utf-8",
            )

            result = validate_dataset(root)

            self.assertEqual(result["synthetic_images"], 1)
            self.assertTrue(any("orphan_label" in value for value in result["invalid_labels"]))

    def test_evaluation_inputs_reject_wrong_class_order_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "test").mkdir(parents=True)
            (root / "data.yaml").write_text(
                "path: .\ntrain: images/test\nval: images/test\ntest: images/test\nnames: [lemon, pumpkin, tomato]\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "class order"):
                validate_evaluation_inputs(root / "data.yaml", manifest)


if __name__ == "__main__":
    unittest.main()
