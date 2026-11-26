"""Hardware-adaptive training entry point for the portable package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    device: str
    imgsz: int
    batch: int
    workers: int
    default_epochs: int


def profile_from_cuda(cuda_available: bool, gpu_name: str | None) -> HardwareProfile:
    if cuda_available:
        return HardwareProfile(
            name=gpu_name or "CUDA GPU",
            device="0",
            imgsz=640,
            batch=2,
            workers=2,
            default_epochs=100,
        )
    return HardwareProfile(
        name="CPU / non-CUDA",
        device="cpu",
        imgsz=416,
        batch=1,
        workers=0,
        default_epochs=10,
    )


def detect_profile() -> HardwareProfile:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch belum terpasang pada interpreter aktif. "
            "Pasang requirements-training.txt terlebih dahulu."
        ) from error
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else None
    return profile_from_cuda(available, name)


def build_command(
    package: Path,
    profile: HardwareProfile,
    *,
    epochs: int,
    name: str,
    batch: int | None = None,
    resume: Path | None = None,
) -> list[str]:
    trainer = package / "training" / "train_tomato_pumpkin_lemon.py"
    data_yaml = package / "dataset" / "data.yaml"
    project = package / "runs" / "adaptive"
    command = [
        sys.executable,
        str(trainer),
        "--data",
        str(data_yaml),
        "--epochs",
        str(epochs),
        "--imgsz",
        str(profile.imgsz),
        "--batch",
        str(batch or profile.batch),
        "--workers",
        str(profile.workers),
        "--device",
        profile.device,
        "--save-period",
        "1",
        "--name",
        name,
        "--project",
        str(project),
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Training adaptive tomato/pumpkin/lemon.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--name")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    try:
        profile = detect_profile()
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 2
    epochs = args.epochs or profile.default_epochs
    name = args.name or f"adaptive-{epochs}ep"
    command = build_command(
        PACKAGE_ROOT,
        profile,
        epochs=epochs,
        name=name,
        batch=args.batch,
        resume=args.resume.resolve() if args.resume else None,
    )
    print(f"Hardware profile: {profile.name}")
    print(f"Training config: epochs={epochs}, imgsz={profile.imgsz}, batch={args.batch or profile.batch}, device={profile.device}, workers={profile.workers}")
    print("Checkpoint: save every epoch to runs/adaptive/<name>/weights/")
    return subprocess.run(command, cwd=PACKAGE_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
