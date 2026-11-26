# SDD ledger — plan: docs/superpowers/plans/2026-11-26-dataset-expansion-tomato-pumpkin-lemon.md

## Pre-flight interface check
Pre-flight: no shared interface conflicts detected.
- Task 1 produces: `data/crops/harvest_manifest.json` and crop files.
- Task 2 consumes: `data/crops/harvest_manifest.json` and crop files; produces: 1,500 synthetic images and labels in `data/processed/tomato-pumpkin-lemon-expanded/`.
- Task 3 consumes: `data/tomato-pumpkin-lemon-training/dataset` (benchmark), harvested crops, and synthetic images; produces: unified expanded dataset with frozen val/test and leakage verification.
- Task 4 consumes: unified expanded dataset; produces: `data/tomato-pumpkin-lemon-expanded.zip` and Colab training script.
- Task 5 verifies: end-to-end smoke test and user guide.

Task 1: complete (commit 4d20243, tests: uv run --with pytest --with pillow pytest tests/test_harvest_crops.py -> 3 passed in 0.37s)
Task 2: complete (commit c243e1a, tests: uv run --with pytest --with pillow pytest tests/test_fruit_composites.py -> 3 passed in 0.62s)
Task 3: complete (commit 4052ce8, tests: uv run --with pytest --with pyyaml --with pillow pytest tests/test_expanded_dataset_integrity.py -> 3 passed in 0.84s)
Task 4: complete (commit a9363ed, tests: uv run --with pytest pytest tests/test_colab_script.py -> 2 passed in 0.29s)
Task 5: complete (commit f0a2e84, tests: uv run --with pytest --with pyyaml --with pillow pytest tests/ -> 13 passed in 1.20s)
Review Remediation: complete (commit bc964fd, tests: uv run --with pytest --with pyyaml --with pillow pytest tests/ -> 15 passed in 0.95s, 0 leaks verified)






