# Adaptive Training Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menambahkan satu runner training yang mendeteksi CUDA/CPU, memilih konfigurasi ringan sesuai perangkat, menyimpan checkpoint, dan dapat melanjutkan training di laptop berbeda.

**Architecture:** `train_auto.py` menjadi entry point lintas laptop dan memanggil trainer paket dengan konfigurasi yang dipilih dari PyTorch. GPU CUDA memakai 640px/batch 2/worker 2, sedangkan CPU/AMD tanpa CUDA memakai 416px/batch 1/worker 0 dan default 10 epoch. Trainer menerima `--save-period` dan `--resume` agar checkpoint `last.pt` dapat dilanjutkan.

**Tech Stack:** Python, PyTorch, Ultralytics YOLO, Windows CMD/VS Code.

**Spec:** `docs/superpowers/plans/2026-11-24-hybrid-training-notebooks.md`

## Global Constraints

- Dataset tetap memakai kelas `0 tomato`, `1 pumpkin`, `2 lemon`.
- `test` dan `external_holdout` tidak digunakan dalam training.
- Training tidak otomatis mengganti model aplikasi.
- Runner tidak memaksa CUDA jika tidak tersedia.
- Checkpoint disimpan oleh Ultralytics dan resume memakai `last.pt` serta `resume=True`.
- Default CPU dibuat ringan agar laptop Vega 8 tidak terbebani berlebihan.
- `.venv` boleh berada di luar folder paket dan interpreter aktif dipakai oleh runner.

## Review Focus

- PyTorch belum terpasang: runner memberi instruksi, bukan traceback yang membingungkan.
- CUDA tidak tersedia: runner memilih CPU tanpa mengirim `--device 0`.
- Resume dari `last.pt`: epoch target tidak menghapus run lama.
- Path paket memiliki spasi atau dipanggil dari folder lain.
- Checkpoint dan hasil tetap berada di folder paket yang dapat dipindahkan.

### Task 1: Tambahkan kontrak runner adaptive

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/tests/test_train_auto.py`

- [ ] **Step 1: Tulis tes konfigurasi CPU/GPU dan command**
- [ ] **Step 2: Jalankan tes untuk memastikan RED**
- [ ] **Step 3: Implementasikan `train_auto.py` dengan `detect_profile()` dan `build_command()`**
- [ ] **Step 4: Jalankan tes dan pastikan GREEN**

### Task 2: Tambahkan checkpoint/resume ke trainer

**Files:**
- Modify: `data/tomato-pumpkin-lemon-training/training/train_tomato_pumpkin_lemon.py`
- Modify: `data/tomato-pumpkin-lemon-training/tests/test_train_auto.py`

- [ ] **Step 1: Tambahkan tes parser untuk `--resume` dan `--save-period`**
- [ ] **Step 2: Jalankan tes RED**
- [ ] **Step 3: Tambahkan argumen trainer dan teruskan ke `model.train()`**
- [ ] **Step 4: Jalankan tes GREEN dan `--validate-only`**

### Task 3: Integrasi VS Code/notebook/README dan uji lokal

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/run_train_auto.cmd`
- Modify: `data/tomato-pumpkin-lemon-training/README_GOOGLE_DRIVE.md`
- Modify: `data/tomato-pumpkin-lemon-training/notebooks/02_train_tomato_pumpkin_lemon.ipynb`
- Modify: `data/tomato-pumpkin-lemon-training/requirements-training.txt`

- [ ] **Step 1: Tambahkan launcher adaptive dan dokumentasi profil**
- [ ] **Step 2: Arahkan notebook ke runner adaptive**
- [ ] **Step 3: Pasang dependency pada interpreter aktif jika tersedia**
- [ ] **Step 4: Jalankan smoke training CPU 1 epoch, tanpa external/test**
- [ ] **Step 5: Pastikan checkpoint/output dan ZIP final bersih**
