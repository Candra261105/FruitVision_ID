# SDD ledger - plan: docs/superpowers/plans/2026-11-24-hybrid-training-notebooks.md
Pre-flight: Task 1 helper/notebook inspection feeds Tasks 2-4. Existing training/evaluation scripts and dataset are present. External .venv support is a documentation/configuration concern, not a package dependency.
Task 1: complete - notebook_helpers.py and 01_dataset_inspection.ipynb added; tests 5 contract checks eventually pass, inspection smoke validation clean.
Task 2: complete - 02_train_tomato_pumpkin_lemon.ipynb added with RTX2050 defaults, CUDA guard, official trainer subprocess, and batch fallback guidance.
Task 3: complete - 03_evaluate_model.ipynb added with best.pt guard, test plus external holdout evaluator command, and per-class display.
Task 4: complete - requirements and README updated for VS Code/Jupyter and external .venv, code cells compile, final ZIP contains exactly 3 notebooks and no best.pt/__pycache__.
Ruling: external .venv is supported by selecting its Python interpreter and Jupyter kernel in VS Code; the package does not create or copy an environment automatically, avoiding duplicate dependencies and large Drive uploads.
