# Hybrid Training Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menambahkan tiga notebook VS Code/Jupyter yang memberi pengalaman seperti `pepper_detection_disease.ipynb`, tetapi memakai path relatif, validasi dataset, trainer resmi, dan evaluasi test/external yang terpisah.

**Architecture:** Notebook hanya menjadi antarmuka interaktif. Notebook inspeksi membaca dataset dan validator paket. Notebook training memeriksa CUDA lalu memanggil `training/train_tomato_pumpkin_lemon.py` melalui subprocess dengan konfigurasi RTX 2050 yang dapat diedit. Notebook evaluasi memanggil evaluator resmi dengan `dataset/data.yaml` untuk test dan `dataset/data-external.yaml` untuk external holdout.

**Tech Stack:** Jupyter Notebook JSON, Python, PyYAML, Pillow, Matplotlib, PyTorch, Ultralytics YOLO.

**Spec:** `docs/superpowers/specs/2026-11-24-tomato-pumpkin-lemon-design.md`

## Global Constraints

- Paket berada di `data/tomato-pumpkin-lemon-training/`.
- Notebook tidak boleh memakai path hard-coded seperti `E:/pepperlens_ai_dataset`.
- Urutan kelas wajib `0 tomato`, `1 pumpkin`, `2 lemon`.
- `train` dipakai untuk training, `val` untuk validasi selama training, `test` untuk pengujian utama setelah training, dan `external_holdout` hanya untuk evaluasi tambahan setelah checkpoint tersedia.
- Komposit sintetis hanya berada di `train`.
- Notebook training tidak boleh otomatis mengganti model aplikasi.
- `best.pt` tidak boleh dianggap tersedia sebelum training nyata selesai.
- Konfigurasi awal RTX 2050 4 GB: `imgsz=640`, `batch=2`, `workers=2`, `device=0`, dengan instruksi fallback `batch=1` untuk CUDA out-of-memory.
- Notebook harus dapat dibuka dan dijalankan dari VS Code setelah folder paket dibuka sebagai workspace.

## Review Focus

- Notebook dijalankan dengan current working directory berbeda tetapi masih di dalam atau di atas folder paket.
- CUDA tidak tersedia atau RTX 2050 tidak terdeteksi, dan notebook berhenti sebelum training CPU tidak sengaja.
- `best.pt` belum ada saat notebook evaluasi dibuka, dan notebook menampilkan instruksi yang jelas tanpa memanggil Ultralytics.
- External holdout tidak ikut masuk ke perintah training atau validasi utama.
- Notebook tidak menyimpan output binary lama yang membuat hasil seolah-olah berasal dari run baru.

### Task 1: Tambahkan helper dan notebook inspeksi dataset

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/notebooks/notebook_helpers.py`
- Create: `data/tomato-pumpkin-lemon-training/notebooks/01_dataset_inspection.ipynb`
- Create: `data/tomato-pumpkin-lemon-training/tests/test_hybrid_notebooks.py`

**Interfaces:**
- Consumes: `dataset/data.yaml`, folder images/labels, dan `training/dataset_validation.py`.
- Produces: helper `find_package_root(start: Path | None = None) -> Path` dan notebook inspeksi tanpa output tersimpan.

- [ ] **Step 1: Tulis tes struktur helper dan notebook**

  Tes harus memastikan root ditemukan dari child path, notebook memakai `dataset/data.yaml`, memuat `tomato`, `pumpkin`, `lemon`, dan memiliki sel validasi serta visualisasi bounding box.

- [ ] **Step 2: Jalankan tes untuk memastikan RED**

  Run: `py data\\tomato-pumpkin-lemon-training\\tests\\test_hybrid_notebooks.py -v`

  Expected: FAIL karena helper/notebook belum ada.

- [ ] **Step 3: Implementasikan helper dan notebook inspeksi**

  Notebook harus menemukan package root dari `Path.cwd()` dengan mencari `dataset/data.yaml`, menampilkan jumlah image/label per split, class counts dari label, beberapa gambar ber-bounding-box, dan hasil `validate_dataset()`.

- [ ] **Step 4: Jalankan tes untuk memastikan GREEN**

  Run: `py data\\tomato-pumpkin-lemon-training\\tests\\test_hybrid_notebooks.py -v`

  Expected: PASS.

### Task 2: Tambahkan notebook training RTX 2050

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/notebooks/02_train_tomato_pumpkin_lemon.ipynb`
- Modify: `data/tomato-pumpkin-lemon-training/tests/test_hybrid_notebooks.py`

**Interfaces:**
- Consumes: `notebook_helpers.find_package_root()` dan trainer paket.
- Produces: notebook dengan konfigurasi `EPOCHS`, `BATCH`, `IMAGE_SIZE`, `DEVICE`, `WORKERS`, pemeriksaan CUDA, dan pemanggilan trainer resmi.

- [ ] **Step 1: Tambahkan tes kontrak training notebook**

  Tes memastikan default `EPOCHS=100`, `BATCH=2`, `IMAGE_SIZE=640`, `DEVICE="0"`, `WORKERS=2`, ada pemeriksaan `torch.cuda.is_available()`, dan command tidak memakai `external_holdout`.

- [ ] **Step 2: Jalankan tes untuk memastikan RED**

  Run: `py data\\tomato-pumpkin-lemon-training\\tests\\test_hybrid_notebooks.py -v`

  Expected: FAIL untuk kontrak notebook training yang belum ada.

- [ ] **Step 3: Implementasikan notebook training**

  Sel pertama menjelaskan dependency dan alur split. Sel konfigurasi dapat diedit. Sel pemeriksaan CUDA menampilkan nama GPU dan berhenti dengan pesan jika CUDA tidak tersedia. Sel training memanggil `training/train_tomato_pumpkin_lemon.py` melalui `sys.executable`, `dataset/data.yaml`, output `runs/notebook-<epoch>ep`, dan tidak membuat evaluasi external.

- [ ] **Step 4: Jalankan tes struktur dan compile code cells**

  Run: `py data\\tomato-pumpkin-lemon-training\\tests\\test_hybrid_notebooks.py -v`

  Expected: PASS dan semua code cell dapat di-compile tanpa menjalankan training.

### Task 3: Tambahkan notebook evaluasi test dan external

**Files:**
- Create: `data/tomato-pumpkin-lemon-training/notebooks/03_evaluate_model.ipynb`
- Modify: `data/tomato-pumpkin-lemon-training/tests/test_hybrid_notebooks.py`

**Interfaces:**
- Consumes: checkpoint `runs/.../weights/best.pt`, evaluator resmi, `dataset/data.yaml`, `dataset/data-external.yaml`, dan `dataset/split_manifest.json`.
- Produces: notebook evaluasi dengan output JSON dan tampilan metrik per kelas.

- [ ] **Step 1: Tambahkan tes kontrak evaluasi**

  Tes memastikan notebook memeriksa keberadaan `best.pt`, memanggil `--data dataset/data.yaml`, `--external-data dataset/data-external.yaml`, dan memakai manifest split. Tes juga memastikan pesan belum ada checkpoint tersedia.

- [ ] **Step 2: Jalankan tes untuk memastikan RED**

  Run: `py data\\tomato-pumpkin-lemon-training\\tests\\test_hybrid_notebooks.py -v`

  Expected: FAIL untuk notebook evaluasi yang belum ada.

- [ ] **Step 3: Implementasikan notebook evaluasi**

  Notebook mencari checkpoint berdasarkan konfigurasi `RUN_DIR`, berhenti jelas jika `best.pt` belum ada, lalu menjalankan evaluator sekali dengan test dan external holdout. Sel berikutnya membaca JSON dan menampilkan status gate serta precision/recall/mAP50/mAP50-95 per kelas.

- [ ] **Step 4: Jalankan tes akhir notebook**

  Run: `py data\\tomato-pumpkin-lemon-training\\tests\\test_hybrid_notebooks.py -v`

  Expected: PASS.

### Task 4: Dokumentasi, dependency notebook, dan paket distribusi

**Files:**
- Modify: `data/tomato-pumpkin-lemon-training/requirements-training.txt`
- Modify: `data/tomato-pumpkin-lemon-training/README_GOOGLE_DRIVE.md`
- Modify: `data/tomato-pumpkin-lemon-training/tests/test_hybrid_notebooks.py`

**Interfaces:**
- Consumes: tiga notebook final dan requirements trainer.
- Produces: instruksi VS Code/Jupyter yang menjelaskan kernel, urutan notebook, konfigurasi RTX 2050, fallback batch, dan batasan evaluasi.

- [ ] **Step 1: Tambahkan dependency notebook minimum**

  Tambahkan `jupyter`, `ipykernel`, `matplotlib`, dan `pillow` dengan batas versi yang konsisten dan tidak menghapus `pyyaml` atau `ultralytics`.

- [ ] **Step 2: Dokumentasikan alur hybrid**

  README harus menjelaskan membuka folder di VS Code, memilih kernel Python, instalasi requirements, menjalankan notebook 01 lalu 02 lalu 03, arti tiap split, dan command fallback `BATCH=1`.

- [ ] **Step 3: Jalankan seluruh pemeriksaan paket**

  Run: `py data\\tomato-pumpkin-lemon-training\\tests\\test_hybrid_notebooks.py -v` dan compile seluruh code cell notebook. Expected: semua tes lulus, tidak ada training yang dijalankan, dan `best.pt` hanya muncul jika pengguna menjalankan notebook training.

- [ ] **Step 4: Perbarui ZIP tanpa membawa output training**

  Hapus `__pycache__` dari paket, pastikan tidak ada `runs/*/weights/best.pt` hasil uji lokal, lalu buat ulang `data/tomato-pumpkin-lemon-training.zip`.
