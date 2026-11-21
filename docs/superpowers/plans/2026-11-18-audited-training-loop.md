# FruitVision ID Audited Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mengunduh sumber object-detection yang diizinkan, menggabungkan anotasi dengan normalisasi kelas konservatif, lalu menjalankan eksperimen detector berulang yang berhenti aman dan memilih model berdasarkan evaluasi yang bebas duplikasi.

**Architecture:** Pipeline dipisah menjadi downloader, auditor/normalizer, preparer YOLO, trainer loop, dan evaluator. Setiap eksperimen memiliki konfigurasi, seed, log metrik, checksum data, dan folder output sendiri. Model hanya dipromosikan jika melewati validasi data, test lintas sumber, dan confidence gate aplikasi.

**Tech Stack:** Python 3.11, Hugging Face `datasets` atau HTTP archive download yang dapat dilanjutkan, Pillow, PyYAML, Ultralytics YOLO, pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-fruitvision-id-design.md`

## Global Constraints

- Top 3 ditentukan dari jumlah bounding box valid gabungan setelah deduplikasi, bukan pilihan manual.
- Normalisasi menyatukan varian ejaan dan bentuk jamak yang jelas, misalnya `Tomato`, `tomatoes`, dan `tomat` menjadi `tomato`.
- `orange`, `lemon`, `lime`, dan `mandarin orange` tetap terpisah.
- `banana` dan `green banana` dihitung terpisah sebelum keputusan final.
- Dataset classification-only tidak masuk detector.
- Dataset dengan lisensi tidak jelas tidak masuk model final.
- Kitchen dipakai sebagai test dunia nyata bila sumber dan anotasinya tersedia, bukan otomatis sebagai training.
- Augmentasi boleh mengubah pencahayaan, kontras, warna sedikit, blur, noise, dan occlusion ringan. Augmentasi tidak boleh mengubah label kelas atau kematangan.
- Tidak ada klaim akurasi sebelum evaluasi selesai.
- Loop dapat berjalan terus, tetapi wajib berhenti saat file `STOP_TRAINING` dibuat, saat storage minimum terlampaui, saat data atau konfigurasi habis, atau saat pengguna menghentikannya.
- Tidak ada unduhan atau training multi-gigabyte tanpa perintah eksplisit pengguna.

## Review Focus

- Dua dataset memiliki gambar yang sama dengan nama file berbeda. Test memastikan hash gambar mencegah kebocoran train/validation/test.
- `tomatoes`, `Tomato`, `tomat`, dan `tomato` muncul sebagai label berbeda. Test memastikan semuanya menjadi `tomato` dan nama asli masuk laporan.
- Box COCO memiliki lebar nol, koordinat keluar gambar, atau gambar tidak ditemukan. Test memastikan box ditolak dan dihitung dalam laporan audit.
- Label kematangan atau penyakit tidak tersedia untuk semua sumber. Test memastikan pipeline tidak mengarang label turunan.
- Proses loop dihentikan di antara dua eksperimen. Test memastikan checkpoint dan hasil eksperimen sebelumnya tetap dapat dilanjutkan.

---

### Task 1: Dataset source registry and resumable download

**Files:**
- Modify: `research/dataset_registry.yaml`
- Create: `research/download_sources.py`
- Create: `research/download_manifest.yaml`
- Test: `backend/tests/test_download_sources.py`

**Interfaces:**
- `download_sources.py --registry research/dataset_registry.yaml --output data/raw --source lvis --source bangladesh --source kitchen`
- `download_sources.download_source(entry: dict, output: Path, allow_unknown_license: bool = False) -> dict`
- Output records URL, license, local path, archive checksum, source version, and download status.

- [ ] Tulis test untuk menolak lisensi `unknown`, melanjutkan file parsial, dan tidak mengunduh dataset classification-only.
- [ ] Jalankan test hingga gagal karena downloader belum ada.
- [ ] Implementasikan sumber yang dapat diverifikasi lebih dulu: LVIS MIT, Bangladesh CC BY 4.0, dan Kitchen CC BY 4.0. Sumber Kaggle harus memakai API atau arsip yang diberikan pengguna, bukan kredensial tertanam.
- [ ] Implementasikan resume, checksum, manifest, dan batas storage minimum.
- [ ] Jalankan test dan `--help`.

### Task 2: Audit and class normalization

**Files:**
- Modify: `research/audit_dataset.py`
- Create: `research/class_normalization.yaml`
- Create: `research/audit_sources.py`
- Modify: `backend/tests/test_research_tools.py`

**Interfaces:**
- `normalize_class_name(raw_name: str, rules: dict) -> str`
- `audit_sources.py --source PATH --source PATH --output research/reports/source-audit.json`
- Report contains valid boxes, invalid boxes, original names, normalized names, image hashes, license status, and source counts.

- [ ] Tulis test untuk contoh `Tomato`, `tomatoes`, `tomat`, `tomato` menjadi satu kelas dan `orange` tetap terpisah dari lemon.
- [ ] Tambahkan test untuk banana versus green banana dan nama kelas yang tidak dikenal.
- [ ] Jalankan test dalam keadaan RED.
- [ ] Implementasikan aturan normalisasi melalui YAML agar dapat diaudit dan tidak tersembunyi dalam kode.
- [ ] Implementasikan deduplikasi lintas sumber dengan SHA-256 dan laporan konflik anotasi.
- [ ] Jalankan test backend dan audit terhadap data lokal yang tersedia.

### Task 3: Combined YOLO preparation and leakage-safe split

**Files:**
- Modify: `research/prepare_yolo.py`
- Create: `research/build_combined_dataset.py`
- Create: `backend/tests/test_combined_dataset.py`

**Interfaces:**
- `build_combined_dataset.py --audit research/reports/source-audit.json --output data/processed/top3 --seed 20261118`
- Output `data.yaml`, `manifest.jsonl`, normalized YOLO labels, class map, source map, and split summary.
- Split is grouped by image hash and source/session group. Kitchen remains test-only unless an explicit flag is used.

- [ ] Tulis test untuk tidak menaruh hash gambar yang sama pada dua split.
- [ ] Tulis test untuk hanya memasukkan tiga kelas hasil ranking dan mempertahankan box valid saja.
- [ ] Jalankan test RED.
- [ ] Implementasikan builder dengan nama file unik, label asli, normalized class ID, dan manifest per gambar.
- [ ] Implementasikan split train/val/test yang deterministic dan seimbang secukupnya per kelas tanpa memindahkan duplikat.
- [ ] Jalankan dry-run dan validasi label.

### Task 4: Augmentation policy and experiment matrix

**Files:**
- Create: `research/augmentation_policy.yaml`
- Create: `research/experiment_matrix.yaml`
- Create: `backend/tests/test_experiment_matrix.py`

**Interfaces:**
- Augmentation groups: original, light-change, dark-change, contrast, mild-color, blur-noise, geometry, and occlusion.
- `experiment_matrix.yaml` enumerates model, image size, epochs, seed, augmentation group, freeze, and patience.
- Validation and test images are never augmented into training copies after splitting.

- [ ] Tulis test untuk menolak augmentation yang mengubah class name atau maturity label.
- [ ] Tulis test untuk memastikan split tetap bebas duplikasi setelah augmentation.
- [ ] Implementasikan konfigurasi eksplisit dan deterministic seed.
- [ ] Jangan membuat augmentasi sintetis tak terbatas. Gunakan batas jumlah dan rasio yang tercatat.
- [ ] Jalankan test dan validasi konfigurasi.

### Task 5: Resumable training loop

**Files:**
- Modify: `research/train_detector.py`
- Create: `research/train_loop.py`
- Create: `research/STOP_TRAINING.example`
- Create: `backend/tests/test_train_loop.py`

**Interfaces:**
- `train_loop.py --data data/processed/top3/data.yaml --matrix research/experiment_matrix.yaml --output runs/fruitvision-loop --device cpu --max-runs 0`
- `max-runs 0` means continue until matrix exhausted, stop file exists, storage guard triggers, or no valid next configuration remains.
- Each run writes `config.json`, `status.json`, training logs, weights, and `metrics.json`.
- A run can be resumed from its status file without retraining completed experiments.

- [ ] Tulis test untuk stop file, resume completed run, and skip invalid configuration.
- [ ] Jalankan test RED.
- [ ] Implementasikan loop sebagai orchestrator, bukan infinite recursion. Poll `STOP_TRAINING` between experiments and epochs where supported.
- [ ] Add CPU-safe defaults for the ThinkPad: small model, bounded image size, early stopping, and one active training process.
- [ ] Do not copy weights into application models automatically.
- [ ] Jalankan test dan `--help`.

### Task 6: Evaluation, model selection, and deployment gate

**Files:**
- Modify: `research/evaluate.py`
- Create: `research/select_model.py`
- Create: `research/reports/README.md`
- Create: `backend/tests/test_model_selection.py`

**Interfaces:**
- Evaluation records mAP50, mAP50-95, precision, recall, per-class recall, crowded subset, Kitchen test metrics, latency, and rejection count.
- `select_model.py --runs runs/fruitvision-loop --output research/reports/model-selection.json`
- Promotion requires data audit pass, no split leakage, valid metrics, and no regression against the baseline on the fixed test set.

- [ ] Tulis test bahwa model dengan mAP tinggi tetapi recall class terendah di bawah gate tidak dipromosikan.
- [ ] Tulis test bahwa model tanpa Kitchen/test result tetap berstatus candidate.
- [ ] Implementasikan ranking model dengan per-class floor, macro metrics, latency, and confidence rejection behavior.
- [ ] Implementasikan copy ke `models/fruit_detector.pt` hanya dengan flag `--promote` setelah gate lulus.
- [ ] Jalankan evaluasi pada semua run yang selesai dan buat laporan jujur.

### Task 7: Health, documentation, and repeatable operation

**Files:**
- Modify: `research/README.md`
- Modify: `README.md`
- Create: `research/run_pipeline.cmd`
- Create: `research/reports/training-loop-report.md`

- [ ] Dokumentasikan perintah unduh, audit, build, training loop, stop, resume, evaluasi, dan promosi.
- [ ] Jalankan command `--help` untuk setiap entry point yang didokumentasikan.
- [ ] Catat ukuran data, lisensi, checksum, jumlah box valid, kelas Top 3, konfigurasi terbaik, dan keterbatasan.
- [ ] Jalankan seluruh test backend, audit, build frontend, dan smoke test sebelum model dipasang ke aplikasi.

## Verification Checklist

- `py -m pytest backend/tests -q`
- Audit setiap sumber dan simpan checksum.
- Pastikan `aggregate.top3` berasal dari box valid, bukan metadata.
- Pastikan hash duplikat tidak menyeberangi split.
- Jalankan loop dengan `--max-runs 1` sebagai smoke test sebelum sesi panjang.
- Buat `STOP_TRAINING` untuk menguji penghentian aman.
- Evaluasi model terbaik pada validation dan Kitchen test.
- Promosi hanya dengan `--promote` dan setelah laporan dapat diperiksa.

## Known Gaps

- Umur dalam hari belum dapat dilatih tanpa data longitudinal.
- Penyakit tidak boleh diprediksi tanpa label penyakit dengan provenance dan definisi yang jelas.
- Ukuran fisik membutuhkan skala kamera atau objek pembanding.
- Tidak ada jaminan akurasi sebelum training dan evaluasi benar-benar selesai.
- Training lokal CPU dapat berlangsung lama dan tidak cocok untuk loop tanpa batas literal.
