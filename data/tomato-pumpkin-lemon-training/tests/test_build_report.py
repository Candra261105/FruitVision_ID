import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "training"))

from dataset_validation import build_dataset


class BuildReportTests(unittest.TestCase):
    def test_build_report_counts_existing_synthetic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_image = root / "source.jpg"
            source_image.write_bytes(b"image")
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({
                "class_names": ["tomato", "pumpkin", "lemon"],
                "records": [{
                    "source_id": "source",
                    "image": str(source_image),
                    "sha256": "source-hash",
                    "phash": "source-phash",
                    "boxes": [{"class_id": 0, "class_name": "tomato", "box": [0.5, 0.5, 0.5, 0.5]}],
                }],
            }), encoding="utf-8")
            output = root / "dataset"
            output.mkdir()
            (output / "images" / "train").mkdir(parents=True)
            (output / "labels" / "train").mkdir(parents=True)
            (output / "images" / "train" / "synthetic.jpg").write_bytes(b"synthetic")
            (output / "labels" / "train" / "synthetic.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (output / "synthetic_manifest.json").write_text(json.dumps({
                "items": [{
                    "output_image": "images/train/synthetic.jpg",
                    "output_label": "labels/train/synthetic.txt",
                    "split": "train",
                    "synthetic": True,
                }]
            }), encoding="utf-8")

            report = build_dataset(audit_path, output)

            self.assertEqual(report["synthetic_images"], 1)

    def test_validation_report_keeps_missing_synthetic_pair_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test", "external_holdout"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            (root / "synthetic_manifest.json").write_text(json.dumps({
                "items": [{
                    "output_image": "images/train/missing.jpg",
                    "output_label": "labels/train/missing.txt",
                    "split": "train",
                    "synthetic": True,
                }]
            }), encoding="utf-8")

            from dataset_validation import validate_dataset
            report = validate_dataset(root)

            self.assertTrue(report["synthetic_errors"])


if __name__ == "__main__":
    unittest.main()
