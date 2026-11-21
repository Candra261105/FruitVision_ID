# YOLO26n Targeted Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menganalisis error baseline YOLO26n epoch 30, memperbaiki train secara terarah, melanjutkan dari bobot yang sudah ada, dan memasang kandidat hanya jika gate evaluasi terpenuhi.

**Architecture:** Analisis baseline berjalan terpisah dari dataset train. Dataset eksperimen baru hanya menyalin train baseline dan mempertahankan val/test/external holdout tanpa perubahan. Training lanjutan dimulai dari `best.pt` epoch 30 sebagai bobot awal pada run baru, lalu dievaluasi dengan test utama yang sama dan external holdout.

**Tech Stack:** Python, Ultralytics YOLO, PyTorch CPU, Pillow, YAML, pytest, FastAPI.

**Spec:** `docs/superpowers/specs/2026-11-21-yolo26n-targeted-improvement-design.md`

## Global Constraints

- YOLO26n tetap dipakai.
- Test utama 320 gambar tidak boleh masuk train atau diubah.
- Validation dan external holdout tetap terpisah.
- Training lanjutan memakai bobot epoch 30, bukan `yolo26n.pt`.
- Model hanya dipasang bila mAP50 `>= 0,55` dan AP50 setiap kelas `>= 0,40`.
- Model gagal gate tidak boleh menggantikan YOLO-World.
- Tidak menambahkan dataset eksternal baru pada putaran ini.

## Review Focus

- Environment global tanpa torch: semua command ML harus memakai `backend/.venv` atau project `.venv` yang terbukti memiliki Ultralytics.
- Checkpoint final tanpa optimizer state: gunakan bobot sebagai initialization pada run baru, bukan mengklaim resume optimizer.
- External holdout hadir tetapi bukan split YAML training: evaluasi dengan `data-external.yaml` dan `split=test` atau konfigurasi yang benar setelah dibaca.
- Analisis error tidak boleh mengubah label test: output hanya laporan JSON/CSV.
- Kandidat rejected tidak boleh tersalin ke `models/`: uji file runtime dan gate sebelum deployment.

---

### Task 1: Verifikasi baseline dan buat analisis error

**Files:**
- Create: `research/analyze_detector_errors.py`
- Create: `runs/loop/yolo26n-error-analysis.json`
- Modify: none in runtime

**Interfaces:**
- CLI `--weights`, `--data`, `--split`, `--output`, `--imgsz`, `--conf`.
- Output JSON berisi total gambar, total ground-truth, total prediksi, per-class counts, false negatives, false positives, class confusion, IoU rendah, dan daftar file bermasalah.

- [ ] Verify `.venv` has torch and ultralytics.
- [ ] Run baseline evaluation on test with existing official evaluator.
- [ ] Run error analysis on the same test without modifying it.
- [ ] Record the dominant class/error pattern and keep the report as provenance.

### Task 2: Build isolated targeted training dataset

**Files:**
- Create: `research/build_targeted_detector_dataset.py`
- Create: `data/processed/combined-top3-cc0-targeted/`
- Create: `data/processed/combined-top3-cc0-targeted/EXPERIMENT_SCOPE.md`

**Interfaces:**
- `build_targeted_dataset(source: Path, output: Path, report: Path) -> dict[str, int]`.
- Source train is copied to the new train split unchanged unless the error report identifies an auditable train-label correction.
- Source val/test/external holdout are copied unchanged.

- [ ] Add a test proving the new dataset does not alter test/val file hashes.
- [ ] Run the test before implementation and confirm failure.
- [ ] Implement isolated copy and manifest generation.
- [ ] Run the focused test and dataset validator.

### Task 3: Run targeted continuation from epoch-30 weights

**Files:**
- Create: `runs/loop/train_yolo26n_targeted.py`
- Create: `runs/detect/runs/loop-detector-yolo26n-targeted/`

**Interfaces:**
- Environment `FRUITVISION_BASE_WEIGHTS` selects the existing epoch-30 `best.pt`.
- Training target is a new run with a bounded continuation budget, preserving the same image size/classes and using the targeted dataset YAML.
- No overwrite of the original run.

- [ ] Verify the base weight hash and target dataset.
- [ ] Start training with `.venv` and a separate log.
- [ ] Monitor checkpoint writes and stop only after the configured continuation completes.
- [ ] Preserve `best.pt`, `last.pt`, args, results, and log.

### Task 4: Evaluate candidate and decide deployment

**Files:**
- Create: `runs/loop/evaluation-yolo26n-targeted-test.json`
- Create: `runs/loop/evaluation-yolo26n-targeted-external.json`
- Create: candidate `.gate.json`
- Modify: `backend/app/config.py` only if gate is eligible
- Copy candidate to `models/` only if gate is eligible

- [ ] Evaluate candidate on unchanged test split.
- [ ] Evaluate candidate on external holdout separately.
- [ ] Compare candidate to baseline and apply all gate thresholds.
- [ ] If rejected, keep `models/yolov8s-world.pt` active and document why.
- [ ] If eligible, copy model and gate metadata, then update runtime configuration.

### Task 5: Verify usable project state

**Files:**
- Modify: `runs/loop/REPORT.md` or create a dated result report.
- Test: backend suite and frontend suite using project commands.

- [ ] Run backend tests.
- [ ] Run frontend tests and production build.
- [ ] Run smoke API and confirm model-info/predict schema.
- [ ] Verify runtime model selection and no rejected model is active.
- [ ] Report factual metrics and any remaining limitations.

## Completion rule

The project is "usable" only if the final runtime model is either an eligible custom model or the existing YOLO-World fallback with explicit demo limitations. A rejected custom model must never be presented as validated.
