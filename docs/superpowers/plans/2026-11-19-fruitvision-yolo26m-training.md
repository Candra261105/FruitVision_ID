# FruitVision YOLO26m Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Melatih dan mengevaluasi YOLO26m custom untuk tiga kelas FruitVision, lalu mempromosikannya hanya jika memenuhi gate metrik yang ditetapkan.

**Architecture:** YOLO26m pretrained dilatih pada dataset bersih `combined-top3-cc0-experiment`. Output training disimpan terpisah di `runs/detect/runs/loop-detector-yolo26m`, dievaluasi dengan provenance SHA-256, lalu registry runtime hanya menerima bobot dengan status `eligible`. YOLO-World tetap menjadi fallback selama kandidat baru belum lolos.

**Tech Stack:** Python 3.11, Ultralytics 8.4.x, PyTorch CPU, pytest, YOLO-format dataset, PowerShell.

**Spec:** `docs/superpowers/specs/2026-11-19-fruitvision-yolo26m-design.md`

## Global Constraints

- Dataset training wajib `data/processed/combined-top3-cc0-experiment/data.yaml`.
- Target wajib tepat tiga kelas: `cucumber`, `tomato`, `green beans`.
- Gate wajib mAP50 keseluruhan `>= 0.55` dan AP50 tiap kelas `>= 0.40`.
- Bobot yang gagal gate tidak boleh disalin ke `models/detector.pt` dan tidak boleh dipakai runtime.
- Training CPU memakai `workers=0`, seed tetap, dan output terpisah dari model aktif.
- External holdout `kitchen-top3` hanya dua kelas, sehingga tidak boleh menjadi bukti gate tiga kelas.
- Tidak ada klaim akurasi berdasarkan smoke test aplikasi.

## Review Focus

- Bobot `yolo26m.pt` tidak tersedia atau gagal dimuat pada versi Ultralytics lokal: training harus berhenti jelas, bukan diam-diam memakai YOLO11/YOLO-World.
- Dataset YAML atau nama kelas berubah: runner harus menolak sebelum training.
- Training berhenti sebelum menghasilkan `best.pt`: hasil harus rejected dan runtime tidak berubah.
- Evaluator hanya menemukan satu atau dua kelas: gate harus rejected walaupun mAP50 terlihat tinggi.
- Bobot eligible rusak atau hash/provenance tidak cocok: registry tidak boleh memuatnya.

---

### Task 1: Tambahkan validasi kandidat dan runner YOLO26m

**Files:**
- Create: `research/yolo26m_training.py`
- Create: `runs/loop/train_yolo26m.py`
- Modify: `backend/tests/test_train_loop.py`

**Interfaces:**
- Produces `validate_detector_dataset(data_yaml: Path) -> dict[str, Any]`.
- Produces `resolve_yolo26m_checkpoint(project: Path, name: str) -> Path`.
- Produces runner yang memanggil `YOLO("yolo26m.pt").train(...)` dengan dataset, `workers=0`, `device="cpu"`, seed `0`, dan output terpisah.

- [ ] **Step 1: Write failing tests**

  Tambahkan test untuk menolak dataset dengan nama kelas tidak tepat, menerima dataset eksperimen bersih, dan menolak checkpoint `best.pt` yang belum ada.

- [ ] **Step 2: Run tests and verify RED**

  Run: `& '.venv\\Scripts\\python.exe' -m pytest backend/tests/test_train_loop.py -q`

  Expected: FAIL karena fungsi validasi dan resolver baru belum tersedia.

- [ ] **Step 3: Implement validation and runner**

  Validasi harus membaca YAML, memeriksa tepat tiga nama kelas, memastikan file dataset ada, dan menggunakan model string `yolo26m.pt` secara eksplisit. Jangan melakukan fallback otomatis ke model lain.

- [ ] **Step 4: Run focused tests and compile**

  Run: `& '.venv\\Scripts\\python.exe' -m pytest backend/tests/test_train_loop.py -q` dan `& '.venv\\Scripts\\python.exe' -m py_compile research/yolo26m_training.py runs/loop/train_yolo26m.py`

  Expected: semua test file lulus dan compile tanpa error.

---

### Task 2: Evaluasi final dan keputusan gate YOLO26m

**Files:**
- Modify: `runs/loop/evaluate_detector_final.py`
- Modify: `backend/tests/test_evaluate.py`
- Create: `runs/loop/yolo26m_gate.json` saat evaluasi nyata

**Interfaces:**
- Memakai `evaluate(weights, data, split, imgsz) -> dict[str, Any]` yang mencatat model, dataset, split, hash bobot, kelas yang ditemukan, metrik per kelas, dan status gate.

- [ ] **Step 1: Write failing tests**

  Tambahkan test untuk kandidat YOLO26m yang memiliki mAP50 tinggi tetapi kehilangan `green beans`, serta test untuk kandidat yang semua kelasnya lengkap namun salah satu AP50 di bawah `0.40`.

- [ ] **Step 2: Run tests and verify RED**

  Run: `& '.venv\\Scripts\\python.exe' -m pytest backend/tests/test_evaluate.py -q`

  Expected: FAIL jika evaluator belum menolak dua kondisi tersebut secara eksplisit.

- [ ] **Step 3: Implement strict gate**

  Status hanya `eligible` bila kelas tepat tiga, mAP50 memenuhi `0.55`, dan seluruh AP50 memenuhi `0.40`. Simpan `weights_sha256` dan provenance dataset.

- [ ] **Step 4: Run focused tests**

  Run: `& '.venv\\Scripts\\python.exe' -m pytest backend/tests/test_evaluate.py -q`

  Expected: semua test lulus.

---

### Task 3: Jalankan training YOLO26m di CPU

**Files:**
- Create: `runs/detect/runs/loop-detector-yolo26m/` sebagai output training
- Create: `runs/loop/yolo26m-training.log`

**Interfaces:**
- Menggunakan runner dari Task 1.
- Checkpoint awal harus `yolo26m.pt` pretrained. Jika Ultralytics mengunduh bobot, URL/status dicatat di log.

- [ ] **Step 1: Preflight tanpa efek runtime**

  Verifikasi proses training lama, ruang disk, keberadaan dataset, dan status model aktif. Jika training lama masih berjalan, hentikan proses training lama sebelum memulai kandidat baru agar CPU tidak berebut.

- [ ] **Step 2: Jalankan training**

  Run: `& '.venv\\Scripts\\python.exe' -u runs/loop/train_yolo26m.py *>> runs/loop/yolo26m-training.log`

  Parameter awal: `imgsz=640`, `batch=2`, `epochs=50`, `patience=10`, `workers=0`, `device=cpu`, `seed=0`, `cache=disk` atau mode yang tidak menghabiskan RAM.

- [ ] **Step 3: Verifikasi checkpoint**

  Expected: `best.pt` ada di output YOLO26m dan ukuran file lebih besar dari nol. Jika tidak ada, tandai rejected dan jangan lanjut promosi.

---

### Task 4: Evaluasi, bandingkan, dan promosi bersyarat

**Files:**
- Modify: `runs/loop/REPORT.md`
- Create: `runs/loop/yolo26m-test-metrics.json`
- Modify: `models/detector.pt` hanya jika gate eligible
- Create: `models/detector.pt.gate.json` hanya jika gate eligible

**Interfaces:**
- Memakai evaluator final dari Task 2 dan checkpoint hasil Task 3.

- [ ] **Step 1: Evaluasi test tiga kelas**

  Run: `& '.venv\\Scripts\\python.exe' runs/loop/evaluate_detector_final.py --weights <best.pt> --data data/processed/combined-top3-cc0-experiment/data.yaml --split test --imgsz 640 --output runs/loop/yolo26m-test-metrics.json`

  Expected: JSON berisi metrik semua tiga kelas dan status gate.

- [ ] **Step 2: Evaluasi holdout kitchen sebagai diagnostik saja**

  Jalankan evaluasi terpisah dan catat bahwa hasilnya hanya mencakup cucumber/tomato, bukan gate tiga kelas.

- [ ] **Step 3: Promosi hanya jika eligible**

  Jika eligible, salin bobot dengan backup file lama dan buat gate manifest. Jika rejected, jangan mengubah `models/detector.pt` dan pertahankan YOLO-World runtime.

- [ ] **Step 4: Jalankan suite lengkap**

  Run: `& '.venv\\Scripts\\python.exe' -m pytest backend/tests -q`, `cmd /c npm test`, dan `cmd /c npm run build`.

  Expected: seluruh backend/frontend test dan build lulus.

- [ ] **Step 5: Smoke test API dan laporan**

  Uji 14 gambar aplikasi, simpan hasil per gambar, dan perbarui `runs/loop/REPORT.md` dengan metrik nyata, hash bobot, status gate, dan batasan. Jangan menyebut smoke test sebagai akurasi.
