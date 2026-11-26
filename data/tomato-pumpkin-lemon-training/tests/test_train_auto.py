import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parents[1]
MODULE_PATH = PACKAGE / "training" / "train_auto.py"
SPEC = importlib.util.spec_from_file_location("train_auto", MODULE_PATH)
train_auto = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_auto
SPEC.loader.exec_module(train_auto)


class TrainAutoTests(unittest.TestCase):
    def test_cpu_profile_is_lightweight(self):
        profile = train_auto.profile_from_cuda(False, None)

        self.assertEqual(profile.device, "cpu")
        self.assertEqual(profile.imgsz, 416)
        self.assertEqual(profile.batch, 1)
        self.assertEqual(profile.workers, 0)
        self.assertEqual(profile.default_epochs, 10)

    def test_cuda_profile_uses_rtx_safe_defaults(self):
        profile = train_auto.profile_from_cuda(True, "NVIDIA GeForce RTX 2050")

        self.assertEqual(profile.device, "0")
        self.assertEqual(profile.imgsz, 640)
        self.assertEqual(profile.batch, 2)
        self.assertEqual(profile.workers, 2)
        self.assertEqual(profile.default_epochs, 100)

    def test_command_never_uses_external_holdout(self):
        profile = train_auto.profile_from_cuda(False, None)
        resume = PACKAGE / "runs" / "adaptive" / "smoke" / "weights" / "last.pt"
        command = train_auto.build_command(
            PACKAGE, profile, epochs=1, name="smoke", resume=resume
        )

        self.assertIn("--data", command)
        self.assertIn(str(PACKAGE / "dataset" / "data.yaml"), command)
        self.assertNotIn("external_holdout", " ".join(command))
        self.assertIn("--save-period", command)
        self.assertIn("--resume", command)

    def test_trainer_exposes_checkpoint_options(self):
        trainer = PACKAGE / "training" / "train_tomato_pumpkin_lemon.py"
        result = subprocess.run(
            [sys.executable, str(trainer), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--save-period", result.stdout)
        self.assertIn("--resume", result.stdout)

    def test_dataset_yaml_resolves_images_from_package_root(self):
        import yaml

        payload = yaml.safe_load((PACKAGE / "dataset" / "data.yaml").read_text(encoding="utf-8"))
        dataset_root = PACKAGE / payload["path"]
        self.assertTrue((dataset_root / payload["train"]).is_dir())
        self.assertTrue((dataset_root / payload["val"]).is_dir())
        self.assertTrue((dataset_root / payload["test"]).is_dir())


if __name__ == "__main__":
    unittest.main()
