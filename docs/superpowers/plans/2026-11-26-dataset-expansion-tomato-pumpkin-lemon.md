# Dataset Expansion & Accuracy Maximization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated dataset expansion and synthetic composite pipeline to elevate FruitVision_ID detection accuracy toward 85%–95%+ for tomato, pumpkin, and lemon, and package it for Google Colab T4 GPU training.

**Architecture:** Automated crop harvester extracts thousands of verified fruit objects from existing repositories (`bangladesh` dataset and `cls_tomato` CC-BY-4.0). A realistic procedural composite engine generates 1,500 rich 800x800 multi-object training scenes (45% lemon, 35% tomato, 20% pumpkin) with natural table/countertop textures and edge feathering. An expanded dataset builder freezes the exact 300 val and 300 test benchmark images, packages everything into `tomato-pumpkin-lemon-expanded.zip`, and prepares an automated Colab training script running YOLO11s at `imgsz=800`.

**Tech Stack:** Python 3.10+, Pillow, NumPy, PyYAML, Ultralytics YOLO11, Pytest.

**Spec:** `docs/superpowers/specs/2026-11-26-dataset-expansion-tomato-pumpkin-lemon-design.md`

## Global Constraints

- Class taxonomy is strictly locked to `0: tomato`, `1: pumpkin`, `2: lemon`.
- No fake/guessed labels or relabeling non-matching fruits (cucumber/limes).
- Zero manual photoshoot requirement: all crops must be automatically harvested from existing verified project data.
- Benchmark validation and test splits must remain strictly frozen (300 val, 300 test) for honest apple-to-apple comparison with the 73.25% baseline.
- Synthetic composites must strictly stay in `train`; `val` and `test` splits must remain 100% real images without pHash/SHA256 leakage.
- Local training cannot run heavy epochs on CPU (Ryzen 3500U, no CUDA); heavy training must be packaged for Google Colab T4 GPU.

## Review Focus

1. Bounding box coordinates in composites exceeding `[0.0, 1.0]` or inverted (`x2 < x1` or `y2 < y1`).
2. Label class ID mismatch during harvest or composite generation (e.g. mapping lemon to 0 instead of 2).
3. Leakage of synthetic images or harvested crop scenes into the frozen `val` or `test` sets.
4. Division by zero or empty bounding boxes for tiny crops smaller than 8px.
5. Incompatible YAML paths in Colab environment (`/content/` vs local absolute paths).

---

### Task 1: Automated Fruit Crop Harvester (`harvest_fruit_crops.py`)

**Files:**
- Create: `research/harvest_fruit_crops.py`
- Test: `tests/test_harvest_crops.py`

**Interfaces:**
- Consumes:
  * `C:\Users\dragon\Downloads\FruitVision_ID\data\processed\bangladesh` (images & labels, class mapping `{14: 'lemon', 16: 'pumpkin', 20: 'tomato'}`)
  * `C:\Users\dragon\Downloads\FruitVision_ID\data\advanced\source\crops\cls_tomato` (pre-cropped tomatoes)
- Produces:
  * Harvested crops directory: `data/crops/{tomato,pumpkin,lemon}/`
  * Manifest file: `data/crops/harvest_manifest.json` with schema `{crop_path, class_id, class_name, width, height, source_image, sha256}`

- [ ] **Step 1: Write the failing test**

In `tests/test_harvest_crops.py`, test:
- Extraction of crops with 4px padding given sample image and YOLO label.
- Enforcement of class taxonomy `{0: 'tomato', 1: 'pumpkin', 2: 'lemon'}`.
- Minimum crop dimension filtering (ignore crops < 12px width or height).
- Manifest output format and SHA256 checksum generation.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_harvest_crops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.harvest_fruit_crops'`

- [ ] **Step 3: Implement `harvest_fruit_crops.py`**

Implement:
- `harvest_from_yolo(image_dir: Path, label_dir: Path, class_map: dict[int, int], output_dir: Path, padding: int = 4) -> list[dict]`
- `ingest_existing_crops(source_dir: Path, class_id: int, class_name: str, output_dir: Path) -> list[dict]`
- `main()` CLI runner with argparse options for output path and minimum crop size.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_harvest_crops.py -v`
Expected: PASS

- [ ] **Step 5: Run harvest on real project data**

Run: `py research/harvest_fruit_crops.py --output data/crops`
Verify that `data/crops/harvest_manifest.json` contains harvested crops for lemon, tomato, and pumpkin.

- [ ] **Step 6: Commit**

```bash
git add research/harvest_fruit_crops.py tests/test_harvest_crops.py
git commit -m "feat(dataset): add automated fruit crop harvester and tests"
```

---

### Task 2: Procedural Background & Realistic Composite Engine (`generate_fruit_composites.py`)

**Files:**
- Create: `research/procedural_backgrounds.py`
- Create: `research/generate_fruit_composites.py`
- Test: `tests/test_fruit_composites.py`

**Interfaces:**
- Consumes:
  * `data/crops/harvest_manifest.json` and crop images from Task 1.
- Produces:
  * 1,500 synthetic composite images in `data/processed/tomato-pumpkin-lemon-expanded/images/train/synthetic_*.jpg`
  * Matching YOLO labels in `data/processed/tomato-pumpkin-lemon-expanded/labels/train/synthetic_*.txt`
  * Manifest file: `data/processed/tomato-pumpkin-lemon-expanded/synthetic_manifest.json`

- [ ] **Step 1: Write the failing test**

In `tests/test_fruit_composites.py`, test:
- Procedural background generator outputs valid RGB image of requested resolution (e.g. 800x800) with realistic texture gradients (wood, marble, countertop).
- Composite generator produces valid YOLO labels where all coordinates `x_center, y_center, width, height` are in `(0.0, 1.0)`.
- Class distribution adheres to target weighting (45% lemon, 35% tomato, 20% pumpkin).
- Edge feathering (Gaussian blend) creates soft alpha mask on crop edges.
- Occlusion scenarios (adjacent, overlapping 10%-35%, clustered, scale jitter).

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_fruit_composites.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.generate_fruit_composites'`

- [ ] **Step 3: Implement `procedural_backgrounds.py` & `generate_fruit_composites.py`**

Implement:
- `generate_texture(width: int, height: int, style: str, rng: random.Random) -> Image.Image` with styles `('wood', 'marble', 'countertop', 'linen')`.
- `apply_feathering(crop: Image.Image, radius: float = 1.0) -> Image.Image`.
- `generate_composites(manifest_path: Path, output_dir: Path, count: int = 1500, canvas_size: tuple[int, int] = (800, 800), seed: int = 20261126) -> dict`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_fruit_composites.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/procedural_backgrounds.py research/generate_fruit_composites.py tests/test_fruit_composites.py
git commit -m "feat(dataset): add procedural backgrounds and realistic composite generator"
```

---

### Task 3: Dataset Integrator & Leakage Guard (`build_expanded_dataset.py`)

**Files:**
- Create: `research/build_expanded_dataset.py`
- Test: `tests/test_expanded_dataset_integrity.py`

**Interfaces:**
- Consumes:
  * Existing benchmark dataset in `data/tomato-pumpkin-lemon-training/dataset`
  * Harvested crops and synthetic composites from Task 1 & Task 2
- Produces:
  * Complete YOLO dataset in `data/processed/tomato-pumpkin-lemon-expanded/` with `images/{train,val,test}` and `labels/{train,val,test}`
  * `data.yaml` configured for YOLO11 training
  * `dataset_report.json` with split counts, class object distributions, and hash audit.

- [ ] **Step 1: Write the failing test**

In `tests/test_expanded_dataset_integrity.py`, test:
- Benchmark `val` (300 images) and `test` (300 images) are byte-for-byte / SHA256 identical to the baseline benchmark.
- Zero pHash/SHA256 leakage between `train` and `val/test`.
- Zero synthetic images in `val` or `test` splits.
- Every label file contains valid class IDs `0, 1, or 2`.
- `data.yaml` has valid relative paths and 3 classes `['tomato', 'pumpkin', 'lemon']`.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_expanded_dataset_integrity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.build_expanded_dataset'`

- [ ] **Step 3: Implement `build_expanded_dataset.py`**

Implement:
- `freeze_benchmark_splits(source_dataset: Path, target_dataset: Path) -> dict`
- `integrate_real_train(source_dataset: Path, target_dataset: Path) -> dict`
- `integrate_synthetic_train(composite_dir: Path, target_dataset: Path) -> dict`
- `verify_leakage(target_dataset: Path) -> bool`
- `write_dataset_yaml(target_dataset: Path) -> Path`
- `main()` orchestrating full dataset build and generating `dataset_report.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_expanded_dataset_integrity.py -v`
Expected: PASS

- [ ] **Step 5: Run full dataset assembly**

Run: `py research/build_expanded_dataset.py --output data/processed/tomato-pumpkin-lemon-expanded`
Verify `data/processed/tomato-pumpkin-lemon-expanded/data.yaml` and dataset report.

- [ ] **Step 6: Commit**

```bash
git add research/build_expanded_dataset.py tests/test_expanded_dataset_integrity.py
git commit -m "feat(dataset): implement expanded dataset builder with strict leakage guard"
```

---

### Task 4: Colab T4 Pipeline Script & Archive Packager

**Files:**
- Create: `research/package_expanded_colab.py`
- Create: `data/tomato-pumpkin-lemon-expanded/training/colab_train_expanded.py`
- Test: `tests/test_colab_script.py`

**Interfaces:**
- Consumes:
  * `data/processed/tomato-pumpkin-lemon-expanded/`
- Produces:
  * `data/tomato-pumpkin-lemon-expanded.zip` ready for upload to Google Colab.
  * Self-contained all-in-one Colab execution script.

- [ ] **Step 1: Write the failing test**

In `tests/test_colab_script.py`, test:
- Syntactic validity and import structure of `colab_train_expanded.py`.
- Correct CLI and training arguments: `imgsz=800`, `batch=32`, `patience=40`, `device=0`, `model='yolo11s.pt'`.
- Verify packaging script bundles dataset, YAML, script, and base model into zip.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_colab_script.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `colab_train_expanded.py` & `package_expanded_colab.py`**

Implement:
- `colab_train_expanded.py` with:
  * Automatic zip extraction if needed.
  * YOLO11s (`yolo11s.pt`) loading and multi-scale 800px training.
  * Post-training evaluation on frozen benchmark `data.yaml`.
  * Class-by-class percentage table display (mAP50, mAP50-95).
  * Inline plotting of `results.png` and `confusion_matrix.png`.
  * Automatic browser download of `best.pt` and `results.png` via `google.colab.files`.
- `package_expanded_colab.py` archiving `data/processed/tomato-pumpkin-lemon-expanded/` and scripts into `data/tomato-pumpkin-lemon-expanded.zip`.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_colab_script.py -v`
Expected: PASS

- [ ] **Step 5: Execute packager**

Run: `py research/package_expanded_colab.py`
Verify that `data/tomato-pumpkin-lemon-expanded.zip` is created.

- [ ] **Step 6: Commit**

```bash
git add research/package_expanded_colab.py data/tomato-pumpkin-lemon-expanded/training/colab_train_expanded.py tests/test_colab_script.py
git commit -m "feat(training): add Colab YOLO11s 800px training runner and zip packager"
```

---

### Task 5: End-to-End Smoke Verification & Deliverable Documentation

**Files:**
- Create: `docs/COLAB_EXPANDED_GUIDE.md`
- Test: `tests/test_end_to_end_smoke.py`

**Interfaces:**
- Consumes:
  * Complete dataset zip `data/tomato-pumpkin-lemon-expanded.zip`
  * `colab_train_expanded.py`
- Produces:
  * Verified end-to-end dry-run report.
  * Clear step-by-step Indonesian guide for the user to run on Google Colab T4.

- [ ] **Step 1: Write smoke verification test**

In `tests/test_end_to_end_smoke.py`, verify:
- Inspect zip archive structure: contains `dataset/data.yaml`, `dataset/images/val`, `dataset/images/train`, `colab_train_expanded.py`.
- Verify total image count (>4,500 images total in train).
- Verify class distribution in labels.

- [ ] **Step 2: Run smoke verification test**

Run: `py -m pytest tests/test_end_to_end_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Create `docs/COLAB_EXPANDED_GUIDE.md`**

Write clear step-by-step instructions in Indonesian:
1. Setting runtime to T4 GPU.
2. Uploading `tomato-pumpkin-lemon-expanded.zip`.
3. Running the 1-cell execution command.
4. Auto-downloading `best.pt` and replacing the local model.

- [ ] **Step 4: Commit**

```bash
git add docs/COLAB_EXPANDED_GUIDE.md tests/test_end_to_end_smoke.py
git commit -m "docs: add Google Colab T4 execution guide and end-to-end smoke test"
```
