# Paket Distribusi Training Tomato Pumpkin Lemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membuat satu folder mandiri di dalam `data` yang dapat diunggah ke Google Drive dan dipakai teman untuk training dataset `tomato`, `pumpkin`, dan `lemon` dengan klik launcher 50–100 epoch.

**Architecture:** Paket distribusi akan menyalin dataset binary final ke subfolder `dataset`, script training/evaluasi ke `training`, dan launcher `.cmd` ke `launchers`. Semua launcher memakai path relatif terhadap lokasi paket, bukan path checkout repository. Folder sumber dataset lama tidak diubah.

**Tech Stack:** Python, Ultralytics YOLO, PyYAML, Windows `.cmd`, PowerShell untuk verifikasi lokal.

**Spec:** `docs/superpowers/specs/2026-11-24-tomato-pumpkin-lemon-design.md`

## Global Constraints

- Paket utama dibuat di `data/tomato-pumpkin-lemon-training/`.
- Kelas dan urutan wajib `0 tomato`, `1 pumpkin`, `2 lemon`.
- Dataset lama tidak boleh ditimpa atau dihapus.
- Komposit sintetis hanya boleh berada pada split `train`.
- Training 50, 60, 70, 80, 90, dan 100 epoch harus tersedia.
- `best.pt` baru tidak boleh diklaim tersedia sebelum training benar-benar dijalankan.
- Paket harus dapat dipindahkan ke Google Drive dan dijalankan dari lokasi baru.
- Jangan memasukkan kredensial, API key, atau file rahasia ke paket.

## Review Focus

- Launcher dijalankan dari folder Drive yang memiliki spasi pada path dan tetap menemukan dataset serta Python.
- Dataset binary lengkap, termasuk `external_holdout`, tidak tertinggal saat penyalinan.
- `data.yaml` memakai path relatif yang benar dari `dataset/`.
- Validasi gagal dengan pesan jelas bila dependency atau file dataset belum tersedia.
- README tidak menjanjikan akurasi dan menyatakan bahwa training belum dijalankan.

### Task 1: Buat struktur paket dan salin dataset

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/dataset/`
- Create: isi dataset dari `data/processed/tomato-pumpkin-lemon/`

**Interfaces:**
- Consumes: dataset final berisi `data.yaml`, `images/`, `labels/`, manifest, laporan, dan README sumber.
- Produces: salinan dataset mandiri dengan kelas dan split yang sama.

- [ ] **Step 1: Buat direktori paket dan subdirektori dataset**

  Buat `data/tomato-pumpkin-lemon-training/dataset/` tanpa menyentuh `data/processed/tomato-pumpkin-lemon/`.

- [ ] **Step 2: Salin seluruh dataset final**

  Salin `images/`, `labels/`, `data.yaml`, `dataset_report.json`, `synthetic_manifest.json`, seluruh laporan validasi, dan `README.md` bila tersedia.

- [ ] **Step 3: Verifikasi isi dan jumlah file**

  Bandingkan jumlah file per split dan ukuran total sumber versus salinan. Pastikan tidak ada file label tanpa pasangan gambar dan tidak ada split sintetis di luar `train`.

### Task 2: Tambahkan script training dan evaluasi mandiri

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/training/train_tomato_pumpkin_lemon.py`
- Create: `data/tomato-pumpkin-lemon-training/training/evaluate_tomato_pumpkin_lemon.py`

**Interfaces:**
- Consumes: `dataset/data.yaml` dan dataset relatif terhadap YAML.
- Produces: command training dengan `--data`, `--epochs`, dan parameter standar, serta evaluasi checkpoint setelah training.

- [ ] **Step 1: Salin trainer dan evaluator dengan import yang tidak bergantung pada root repository**

  Trainer paket harus dapat dipanggil langsung dari `training/`, memakai `dataset/data.yaml`, dan tidak mengimpor modul `research` yang tidak ikut paket. Pertahankan validasi kelas, split, label, dan duplicate lintas split.

- [ ] **Step 2: Tetapkan output training di dalam paket**

  Default output diarahkan ke `runs/` di root paket agar hasil dapat ikut disimpan atau dikirim kembali tanpa mengubah repository utama. `best.pt` hanya dibuat oleh proses training nyata.

- [ ] **Step 3: Jalankan validasi ringan**

  Jalankan trainer dengan `--validate-only --epochs 1` memakai Python yang tersedia. Expected: validasi berhasil dan menampilkan kelas `tomato`, `pumpkin`, `lemon` tanpa membuat checkpoint.

### Task 3: Buat launcher klik dua kali

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/launchers/train_50_epochs.cmd`
- Create: `data/tomato-pumpkin-lemon-training/launchers/train_60_epochs.cmd`
- Create: `data/tomato-pumpkin-lemon-training/launchers/train_70_epochs.cmd`
- Create: `data/tomato-pumpkin-lemon-training/launchers/train_80_epochs.cmd`
- Create: `data/tomato-pumpkin-lemon-training/launchers/train_90_epochs.cmd`
- Create: `data/tomato-pumpkin-lemon-training/launchers/train_100_epochs.cmd`

**Interfaces:**
- Consumes: `training/train_tomato_pumpkin_lemon.py`, `dataset/data.yaml`, dan Python environment lokal.
- Produces: enam launcher Windows dengan epoch tetap dan path paket relatif.

- [ ] **Step 1: Buat template launcher dengan root paket dari `%~dp0`**

  Launcher harus berpindah ke root paket, mencoba `.venv\\Scripts\\python.exe` bila tersedia, memberi pesan bila Python/dependency tidak ada, lalu memanggil trainer dengan `--data dataset\\data.yaml --epochs N`.

- [ ] **Step 2: Buat enam preset epoch**

  Gunakan nilai tepat 50, 60, 70, 80, 90, dan 100. Setiap launcher menulis hasil ke `runs/` melalui default trainer.

- [ ] **Step 3: Uji path launcher secara statis**

  Jalankan pemeriksaan PowerShell terhadap keenam file dan pastikan semua memuat path `dataset\\data.yaml`, script `training\\train_tomato_pumpkin_lemon.py`, serta epoch yang sesuai.

### Task 4: Dokumentasi Google Drive dan pemeriksaan akhir

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/README_GOOGLE_DRIVE.md`
- Create: `data/tomato-pumpkin-lemon-training/requirements-training.txt`

**Interfaces:**
- Consumes: struktur final paket dan kebutuhan import trainer/evaluator.
- Produces: instruksi yang dapat diikuti teman tanpa mengetahui struktur repository asli.

- [ ] **Step 1: Tulis README penggunaan**

  Jelaskan cara mengunduh satu folder, memasang Python/dependency, membuat `.venv` opsional, memilih launcher, lokasi output, estimasi bahwa training dapat lama, dan fakta bahwa model baru belum tersedia sebelum training selesai.

- [ ] **Step 2: Tulis daftar dependency minimum**

  Cantumkan dependency yang benar-benar dipakai trainer/evaluator, termasuk versi minimum yang konsisten dengan project.

- [ ] **Step 3: Verifikasi penerimaan paket**

  Jalankan pemeriksaan struktur, `data.yaml`, jumlah split, validasi dataset, dan keenam launcher. Expected: semua pemeriksaan lulus, dataset sumber tetap utuh, dan tidak ada `best.pt` yang diklaim atau dibuat oleh pemeriksaan ringan.

- [ ] **Step 4: Buat ZIP opsional tanpa menghapus folder utama**

  Setelah folder lulus pemeriksaan, buat salinan ZIP di dalam `data/` atau lokasi yang jelas. Folder utama tetap menjadi paket yang direkomendasikan untuk diunggah ke Drive.
