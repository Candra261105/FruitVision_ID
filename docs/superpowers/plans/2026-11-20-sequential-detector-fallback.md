# Sequential Detector Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menjalankan detector custom yang sudah lolos gate terlebih dahulu, lalu memakai YOLO-World sebagai fallback berurutan tanpa mencampur bounding box kedua model.

**Architecture:** Registry memuat runtime primary dan fallback secara terpisah. Sebuah provider berurutan menjalankan primary, mengembalikan hasilnya bila ada, dan hanya memanggil fallback ketika primary tidak tersedia, kosong, atau error. Setiap `RawDetection` membawa sumber detector sampai ke response API.

**Tech Stack:** Python, FastAPI, Pydantic, Pillow, Ultralytics, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-sequential-detector-fallback-design.md`

## Global Constraints

- Detector custom hanya dipakai jika file `.gate.json` memiliki `status` `eligible`.
- YOLO-World adalah fallback demo, bukan bukti detector custom tervalidasi.
- Tidak ada penggabungan bounding box pada tahap ini.
- Validation/test utama dan external holdout tidak diubah.
- Model gagal gate tidak boleh dipasang sebagai detector utama.
- Hasil confidence rendah tetap berstatus `review`.
- Analisis maturity, rot, disease, dan grade tetap berjalan dari crop bounding box akhir.
- Tidak boleh mengklaim akurasi meningkat sebelum evaluasi ulang.

## Review Focus

- Primary path hilang atau gate tidak eligible: fallback tetap harus dimuat dan aplikasi tidak boleh mengeluarkan status custom.
- Primary menghasilkan list kosong: fallback dipanggil tepat sekali dan hasilnya tetap memiliki sumber `world_fallback`.
- Primary melempar exception: request tidak gagal jika fallback berhasil.
- Kedua model menghasilkan list kosong: response tetap valid dengan nol deteksi dan status model yang benar.
- Provider lama atau fake provider tidak mengisi sumber: wrapper memberi sumber berdasarkan jalur yang digunakan tanpa merusak test/provider lama.

---

### Task 1: Tambahkan metadata sumber pada domain deteksi

**Files:**
- Modify: `backend/app/inference/base.py`
- Modify: `backend/app/schemas/prediction.py`
- Modify: `backend/app/services/prediction_service.py`
- Test: `backend/tests/test_prediction_domain.py`

**Interfaces:**
- `RawDetection.detector_source: str | None = None` untuk kompatibilitas provider lama.
- `Detection.detector_source: Literal["custom", "world_fallback"]`, default ditentukan dari mode bila raw source kosong.

- [ ] **Step 1: Write failing tests**

Tambahkan test yang membuat `RawDetection(..., detector_source="world_fallback")`, menjalankan `PredictionService`, lalu memastikan `result.detections[0].detector_source == "world_fallback"`. Tambahkan test kompatibilitas tanpa source yang memastikan mode `custom` menghasilkan `custom`.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `py -m pytest backend/tests/test_prediction_domain.py -k "detector_source" -v`

Expected: FAIL karena field metadata belum tersedia atau belum diteruskan.

- [ ] **Step 3: Implement metadata propagation**

Tambahkan field default pada `RawDetection`, field tervalidasi pada schema `Detection`, dan teruskan source saat `PredictionService.predict()` membangun `Detection`. Gunakan `raw.detector_source` bila ada, jika tidak gunakan `world_fallback` untuk mode demo dan `custom` untuk mode custom/test.

- [ ] **Step 4: Run focused tests to verify pass**

Run: `py -m pytest backend/tests/test_prediction_domain.py -k "detector_source" -v`

Expected: seluruh test metadata lulus.

- [ ] **Step 5: Run existing domain tests**

Run: `py -m pytest backend/tests/test_prediction_domain.py -v`

Expected: seluruh test domain lulus tanpa perubahan perilaku lama.

### Task 2: Beri label sumber pada Ultralytics provider dan buat provider berurutan

**Files:**
- Modify: `backend/app/inference/ultralytics_provider.py`
- Create: `backend/app/inference/sequential_provider.py`
- Test: `backend/tests/test_ultralytics_provider.py`
- Create: `backend/tests/test_sequential_provider.py`

**Interfaces:**
- `UltralyticsProvider(..., detector_source: str = "custom") -> InferenceProvider`.
- `SequentialDetectorProvider(primary: InferenceProvider | None, fallback: InferenceProvider | None) -> InferenceProvider`.
- `SequentialDetectorProvider.__call__(image: Image.Image) -> list[RawDetection]`.

- [ ] **Step 1: Write failing provider tests**

Buat fake primary/fallback dengan penghitung panggilan. Uji primary berhasil tidak memanggil fallback, primary kosong memanggil fallback, primary error memanggil fallback, dan dua-duanya kosong menghasilkan list kosong. Pastikan hasil fallback memiliki `detector_source == "world_fallback"`, termasuk saat fake provider tidak mengisi source.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `py -m pytest backend/tests/test_sequential_provider.py -v`

Expected: FAIL karena module/provider baru belum ada.

- [ ] **Step 3: Implement sequential provider**

Jalankan primary jika tersedia dalam `try/except Exception`. Kembalikan hasil primary jika tidak kosong. Jika kosong atau exception, jalankan fallback jika tersedia. Gunakan `dataclasses.replace` untuk mengisi source yang kosong, dan jangan mengubah source yang sudah diisi. Jika fallback gagal, kembalikan list kosong agar request tidak jatuh.

- [ ] **Step 4: Add source to Ultralytics detections**

Tambahkan parameter `detector_source` pada `UltralyticsProvider` dan isi field source pada semua jalur `_predict`, retry context, dan tiling melalui `RawDetection`.

- [ ] **Step 5: Run focused tests to verify pass**

Run: `py -m pytest backend/tests/test_sequential_provider.py backend/tests/test_ultralytics_provider.py -v`

Expected: seluruh test provider lulus.

### Task 3: Ubah registry dan konfigurasi menjadi primary + fallback

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/services/model_registry.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_ultralytics_provider.py`
- Create or modify: test konfigurasi/registry yang sudah ada bila ditemukan saat implementasi

**Interfaces:**
- `Settings.primary_model_path: Path`.
- `Settings.fallback_model_path: Path`.
- `load_runtime(primary_model_path: Path, confidence_threshold: float, fallback_model_path: Path | None = None) -> ModelRuntime`.

- [ ] **Step 1: Write failing registry tests**

Tambahkan test untuk: primary eligible + fallback tersedia menghasilkan mode `custom` dan provider sequential, primary tidak eligible + fallback tersedia menghasilkan mode `demo`, dan primary hilang + fallback tersedia tetap menghasilkan runtime model loaded. Tambahkan test bahwa path fallback lama tetap dapat dipakai ketika `fallback_model_path` tidak diberikan.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `py -m pytest backend/tests/test_ultralytics_provider.py -k "fallback or primary" -v`

Expected: FAIL karena registry masih memuat satu model.

- [ ] **Step 3: Implement configuration compatibility**

Tambahkan `FRUITVISION_PRIMARY_MODEL_PATH` dengan default `models/detector.pt` dan `FRUITVISION_FALLBACK_MODEL_PATH` dengan default `models/yolov8s-world.pt`. Pertahankan `FRUITVISION_MODEL_PATH` sebagai alias primary bila env baru tidak diset, supaya konfigurasi lama tidak diam-diam berubah.

- [ ] **Step 4: Implement registry loading rules**

Pisahkan helper pemuatan satu model dari `load_runtime`. Primary hanya dibuat bila file ada dan gate eligible. Fallback hanya dibuat bila file ada dan dimuat sebagai YOLO-World dengan `detector_source="world_fallback"`. Gabungkan keduanya melalui `SequentialDetectorProvider`. Runtime mode `custom` bila primary aktif, `demo` bila hanya fallback aktif, dan `unavailable` bila keduanya tidak aktif. Limitation harus menyebut fallback demo bila primary tidak aktif.

- [ ] **Step 5: Wire settings through application startup**

Ubah `main.py` agar memanggil `load_runtime(settings.primary_model_path, settings.confidence_threshold, settings.fallback_model_path)`. Pastikan `advanced_models` tetap dimuat dari direktori model yang sama dan tidak terkena perubahan.

- [ ] **Step 6: Run registry/config tests**

Run: `py -m pytest backend/tests/test_ultralytics_provider.py backend/tests/test_model_selection.py -v`

Expected: seluruh test registry dan model selection lulus.

### Task 4: Verifikasi integrasi aplikasi dan dokumentasi runtime

**Files:**
- Modify: `backend/app/services/model_registry.py` bila `model_info()` perlu menampilkan status primary/fallback.
- Modify: `backend/app/schemas/prediction.py` bila metadata model perlu diperluas.
- Modify: `runs/loop/REPORT.md` dengan hasil verifikasi aktual.
- Test: suite backend yang sudah ada.

**Interfaces:**
- Endpoint `/api/model-info` tetap kompatibel dan, bila ditambah, menyediakan status sumber runtime tanpa menghapus field lama.
- Endpoint `/api/predict/image` mengembalikan `detector_source` pada setiap detection.

- [ ] **Step 1: Run complete backend tests**

Run: `py -m pytest backend/tests -q`

Expected: exit code 0 dan tidak ada test gagal.

- [ ] **Step 2: Run frontend tests and production build**

Run dari direktori frontend menggunakan perintah yang sudah dipakai proyek.

Expected: test frontend lulus dan production build exit code 0.

- [ ] **Step 3: Run smoke API**

Jalankan smoke test aplikasi pada kumpulan gambar yang sudah digunakan proyek.

Expected: seluruh request memperoleh HTTP 200, response tetap memenuhi schema, dan deteksi jika ada memiliki `detector_source` valid.

- [ ] **Step 4: Verify model gate and runtime selection**

Pastikan file custom yang belum `eligible` tidak dipakai sebagai primary. Pastikan konfigurasi default tanpa custom eligible tetap menggunakan `models/yolov8s-world.pt` sebagai demo fallback. Jangan menyalin model gagal ke registry runtime.

- [ ] **Step 5: Record factual results**

Perbarui `runs/loop/REPORT.md` dengan file yang berubah, hasil test, mode runtime, dan batasan. Tulis tegas bahwa sequential fallback adalah perubahan alur runtime, bukan bukti peningkatan mAP.

## Self-review hasil perencanaan

- Semua bagian spec tercakup: loading dua runtime, urutan primary/fallback, source metadata, kondisi error, dan verifikasi tanpa mengubah split.
- Tidak ada task yang mempromosikan model gagal gate.
- Interface antar-task konsisten: `RawDetection.detector_source` diteruskan oleh provider ke `Detection.detector_source`, sedangkan registry menyuplai `SequentialDetectorProvider` ke `PredictionService`.
- Lima failure mode pada Review Focus memiliki test pada Task 1–3.
- Evaluasi mAP tidak dijalankan sebagai bagian implementasi runtime karena perubahan ini tidak mengubah bobot atau dataset. Evaluasi tetap wajib bila model custom baru akan dipasang.
