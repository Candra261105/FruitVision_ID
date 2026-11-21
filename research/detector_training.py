from __future__ import annotations

from pathlib import Path


def find_best_weight(output_dir: Path, run_name: str, project_root: Path) -> Path:
    """Find the checkpoint written by Ultralytics for one detector run.

    Ultralytics may resolve a relative ``project`` below its own ``runs``
    directory. Keep the direct path first, then check the known nested
    ``runs/detect/runs`` layout. The run name is always part of the path so a
    checkpoint from another experiment cannot be selected accidentally.
    """
    candidates = (
        output_dir / run_name / "weights" / "best.pt",
        project_root / "runs" / "detect" / output_dir.name / run_name / "weights" / "best.pt",
        project_root / "runs" / "detect" / output_dir / run_name / "weights" / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    nested = sorted(
        path
        for path in (project_root / "runs").glob("**/" + run_name + "/weights/best.pt")
        if path.is_file() and path.parent.parent.name == run_name
    )
    if len(nested) == 1:
        return nested[0].resolve()
    if len(nested) > 1:
        raise FileExistsError(
            f"Beberapa best.pt ditemukan untuk run {run_name!r}: "
            + ", ".join(str(path) for path in nested)
        )
    raise FileNotFoundError(
        f"best.pt untuk run {run_name!r} tidak ditemukan di {output_dir} "
        f"atau {project_root / 'runs'}"
    )
