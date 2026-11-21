# FruitVision Inspection Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menjalankan loop training detector FruitVision secara resumable dan menyiapkan jalur evaluasi yang tidak mengaktifkan klaim penyakit, kematangan, busuk, atau grade tanpa dataset berlabel yang sesuai.

**Architecture:** Dataset final tiga kelas tetap menjadi sumber training. Matrix menjalankan tiga konfigurasi YOLO11n berurutan, menyimpan checkpoint dan metrik terstruktur, lalu model diuji pada validation dan Kitchen sebelum gate promosi. Fitur tambahan hanya boleh diaktifkan sebagai model terpisah setelah data dan evaluasinya tersedia.

**Tech Stack:** Python, Ultralytics YOLO, PyTorch CPU, pytest, YAML, FastAPI/React yang sudah ada.

**Spec:** `docs/superpowers/specs/2026-11-18-fruitvision-inspection-design.md`

## Global Constraints

- Kelas detector saat ini hanya `cucumber`, `tomato`, dan `green beans`.
- Kitchen tetap menjadi test lintas-domain dan tidak masuk training.
- Confidence rendah harus ditolak atau diberi status review.
- Penyakit, umur hari, kematangan, busuk, ukuran fisik, dan grade tidak boleh diklaim tanpa label/provenance/kalibrasi yang sesuai.
- API key provider AI hanya boleh dibaca backend dari environment.
- Model tidak dipromosikan atau dipasang ke aplikasi sebelum lolos gate evaluasi.
- Training dapat dihentikan dengan file `STOP_TRAINING` dan dilanjutkan dari `weights/last.pt`.

## Review Focus

- Run terhenti setelah checkpoint: harus resume, bukan mulai dari epoch nol. Test: `test_find_resume_checkpoint_finds_last_checkpoint_inside_run`.
- Matrix memiliki model lokal atau perlu download: status dan log harus jelas. Test: matrix validation dan smoke run.
- Kitchen tidak memiliki `green beans`: metrik harus menyatakan kelas tidak tersedia, bukan nol diam-diam. Test: evaluator per-class output.
- Model gagal gate tetapi mAP cukup tinggi: tetap rejected karena recall per kelas. Test: `test_select_model_rejects_high_map_with_low_class_recall`.
- Fitur tambahan tanpa dataset: harus berstatus belum tersedia. Test: model capability contract pada tahap integrasi web.

---

### Task 1: Verifikasi dataset dan matrix training

**Files:**
- Read: `data/processed/combined-top3-final/data.yaml`
- Read: `research/experiment_matrix.yaml`
- Read: `research/train_loop.py`
- Output: `runs/fruitvision-loop`

**Interfaces:**
- Consumes: dataset final, matrix tiga eksperimen, `STOP_TRAINING`.
- Produces: `status.json`, `weights/last.pt`, `weights/best.pt`, `metrics.json` per run.

- [ ] Hitung gambar per kelas dan verifikasi 3.191 gambar, 0 hash leakage, dan 0 label error.
- [ ] Pastikan `yolo11n.pt` tersedia lokal atau catat download Ultralytics di log.
- [ ] Jalankan `train_loop.py` tanpa `--max-runs` di CPU sebagai background process.
- [ ] Jangan menjalankan proses kedua jika `runs/fruitvision-loop` masih aktif.

### Task 2: Evaluasi setiap eksperimen

**Files:**
- Modify: `research/evaluate.py` bila metrik per kelas belum lengkap.
- Output: `research/reports/evaluation-*.json`

**Interfaces:**
- Consumes: `weights/best.pt` dan `data/processed/combined-top3-final/data.yaml` atau `data/processed/kitchen-top3/data.yaml`.
- Produces: mAP, precision, recall, dan metrik per kelas.

- [ ] Evaluasi validation setiap run yang completed.
- [ ] Evaluasi Kitchen 120 gambar setiap run yang completed.
- [ ] Catat kelas Kitchen yang tidak memiliki anotasi sebagai unavailable.

### Task 3: Gate promosi model

**Files:**
- Read: `research/select_model.py`
- Output: `research/reports/model-selection.json`

**Interfaces:**
- Consumes: `metrics.json`, metrik Kitchen, ambang recall minimum 0,3.
- Produces: `rejected`, `candidate`, atau `eligible` dan `best_eligible`.

- [ ] Jalankan gate untuk semua run.
- [ ] Promosikan hanya jika ada `eligible` dan semua syarat lintas-domain terpenuhi.
- [ ] Jangan membuat `models/fruit_detector.pt` dari smoke atau run rejected.

### Task 4: Model tambahan yang sah

**Files:**
- Create later: dataset manifests dan model-specific training configs.
- Modify later: model capability schema dan web status panel.

**Interfaces:**
- Consumes: dataset berlabel maturity, rot, disease, dan grade dengan provenance.
- Produces: model terpisah per task, bukan satu model campuran tanpa label.

- [ ] Audit ketersediaan dataset khusus sebelum membuat training loop tambahan.
- [ ] Jika dataset belum tersedia, tulis status `unavailable` dan rekomendasi pengumpulan data.
- [ ] Jangan memakai AI provider untuk mengisi label training atau menggantikan validasi ahli.

### Task 5: Verification

- [ ] Jalankan `py -m pytest backend/tests -q`.
- [ ] Periksa semua `status.json` dan proses yang tersisa.
- [ ] Periksa gate model dan pastikan tidak ada promosi palsu.
- [ ] Laporan akhir menyebutkan run selesai, run terhenti, metrik, dan keterbatasan.
