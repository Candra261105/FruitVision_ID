# 35-Class Fruit & Vegetable Production Model Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the verified 35-class Hugging Face YOLO11m detector into FruitVision ID backend and frontend, localize all class names, remove legacy Google Drive links, and push the completed production-ready application to GitHub.

**Architecture:** Configure `models/fruit_veg_yolo11m.pt` with its matching `gate.json` descriptor as default model. Expand backend `model_registry` and `prediction_service` to dynamically recognize and Indonesian-localize all 35 fruits and vegetables. Update frontend `App.tsx` and strip Google Drive references from `README.md`.

**Tech Stack:** Python 3.11, FastAPI, Ultralytics YOLO11, React, TypeScript, Vite.

**Spec:** `docs/superpowers/specs/2026-11-27-fruit-veg-35-class-production-design.md`

## Global Constraints
- Python runtime must execute on CPU (Ryzen 3500U) with fast inference (<1s per image).
- Maintain 100% backward compatibility with existing API endpoints (`/health`, `/api/model-info`, `/api/predict/image`).
- No git tracking of binary models >50MB without verification (current model is 38.7 MB, and `.gitignore` ignores `models/*.pt`).

## Review Focus
1. Model loading without gate failure when `.gate.json` specifies 35 classes.
2. Proper detection mapping for all 35 fruit & vegetable classes with Indonesian labels.
3. Fallback downloader if model file is not found locally.
4. Clean documentation with all Google Drive links removed.
5. All backend and root tests passing.

---

### Task 1: Create Gate File and Update Backend Config for 35-Class Detector

**Files:**
- Create: `models/fruit_veg_yolo11m.gate.json`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_model_selection.py`

**Interfaces:**
- Consumes: `models/fruit_veg_yolo11m.pt`
- Produces: `Settings.model_path` pointing to `fruit_veg_yolo11m.pt` when present

- [ ] **Step 1: Write gate descriptor `models/fruit_veg_yolo11m.gate.json`**
Status `eligible`, 35 classes, mAP50 0.925.

- [ ] **Step 2: Update `backend/app/config.py` default model path**
Check if `root / "models" / "fruit_veg_yolo11m.pt"` exists and use it as default when `FRUITVISION_MODEL_PATH` is not explicitly set.

- [ ] **Step 3: Run model selection tests**
Run: `uv run pytest backend/tests/test_model_selection.py -v`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add models/fruit_veg_yolo11m.gate.json backend/app/config.py
git commit -m "feat(backend): configure 35-class fruit_veg_yolo11m as default production model"
```

---

### Task 2: Expand Backend Model Registry & Prediction Service for 35 Classes

**Files:**
- Modify: `backend/app/services/model_registry.py`
- Modify: `backend/app/services/prediction_service.py`
- Create: `backend/tests/test_fruit_veg_35_classes.py`

**Interfaces:**
- Consumes: 35 classes from `UltralyticsProvider`
- Produces: Localized Indonesian fruit names for all 35 classes in `PredictionResponse`

- [ ] **Step 1: Write test for 35-class detection and localization**
Verify that all 35 classes are recognized by `load_runtime` and translated into Indonesian in `PredictionService`.

- [ ] **Step 2: Update `model_registry.py`**
Allow gate classes matching any valid model gate (including 35 classes) and pass all model classes to `ModelRuntime`.

- [ ] **Step 3: Update `prediction_service.py`**
Add comprehensive Indonesian dictionary for all 35 classes (Apple -> Apel, Banana -> Pisang, Orange -> Jeruk, Strawberry -> Stroberi, Tomato -> Tomat, etc.) and fallback gracefully for any unknown label.

- [ ] **Step 4: Run tests**
Run: `uv run pytest backend/tests/test_fruit_veg_35_classes.py backend/tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add backend/app/services/model_registry.py backend/app/services/prediction_service.py backend/tests/test_fruit_veg_35_classes.py
git commit -m "feat(backend): support 35 fruit and vegetable classes with Indonesian localization"
```

---

### Task 3: Add Model Auto-Downloader Utility

**Files:**
- Create: `research/download_production_model.py`
- Test: `tests/test_production_model_downloader.py`

**Interfaces:**
- Produces: `ensure_production_model(dest: Path) -> Path` downloading from Hugging Face if not present

- [ ] **Step 1: Write test for model downloader**
- [ ] **Step 2: Implement `download_production_model.py`**
- [ ] **Step 3: Run downloader test**
- [ ] **Step 4: Commit**
```bash
git add research/download_production_model.py tests/test_production_model_downloader.py
git commit -m "feat: add production model auto-downloader from huggingface"
```

---

### Task 4: Update Frontend App and UI for 35 Classes

**Files:**
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `model.classes` from `/api/model-info`
- Produces: Dynamic commodity strip displaying active classes or multi-class production badge

- [ ] **Step 1: Update `frontend/src/App.tsx` intro text and target badge**
Show dynamic class count (e.g. "35 Jenis Buah & Sayur") and render top popular target tags (Apel, Pisang, Jeruk, Tomat, Mangga, Stroberi, dll.) dynamically from model info.

- [ ] **Step 2: Verify frontend builds without errors**
Run: `npm run build` inside `frontend/`

- [ ] **Step 3: Commit**
```bash
git add frontend/src/App.tsx
git commit -m "feat(frontend): display 35-class production fruit & vegetable support"
```

---

### Task 5: Clean Documentation, Remove Google Drive Links, and Final Push

**Files:**
- Modify: `README.md`
- Modify: `data/README.md`

**Interfaces:**
- Produces: Updated documentation referencing the production-ready 35-class model and removing dead Drive links

- [ ] **Step 1: Remove all Google Drive links from `README.md` and `data/README.md`**
Replace with direct production model usage and Colab cloud training guide.

- [ ] **Step 2: Run all backend and root test suites**
Run: `uv run pytest backend/tests/` and `$env:PYTHONPATH="."; uv run pytest tests/`
Expected: All tests PASS.

- [ ] **Step 3: Commit and Push to GitHub**
```bash
git add README.md data/README.md
git commit -m "docs: finalize production release with 35-class model and remove drive links"
git push origin main
```
