# FruitVision ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Membuat MVP offline yang bisa menerima foto/preview kamera, mendeteksi lima buah prioritas dengan bounding box, mengklasifikasikan tahap kematangan bila bobot tersedia, dan menampilkan status yang jujur bila fitur belum terkalibrasi.

**Architecture:** Backend FastAPI memisahkan decode gambar, inference provider, normalisasi response, dan aturan status. Frontend React + TypeScript mengambil snapshot kamera atau file lalu menggambar bounding box di canvas. Registry dan skrip audit dataset disiapkan agar bobot pretrained dapat diganti dengan bobot hasil training tanpa mengubah API.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, OpenCV/Pillow, Ultralytics sebagai provider detector opsional, React, TypeScript, Vite, CSS biasa, pytest, Playwright smoke test.

**Spec:** `docs/superpowers/specs/2026-11-18-fruitvision-id-design.md`

## Global Constraints

- MVP wajib mendeteksi jenis, jumlah, posisi, buah berdempetan, dan kematangan jika labelnya tersedia.
- Bounding box wajib, mask opsional.
- Buah prioritas: pisang, mangga, pepaya, alpukat, jeruk.
- Kualitas, penyakit, dan masalah visual ditunda dari MVP.
- Input di luar kelas ditolak dengan peringatan confidence rendah.
- Target inferensi snapshot maksimal lima detik pada laptop lokal.
- Aplikasi harus berjalan offline setelah model dan dependensi dipasang.
- Dataset diunduh bertahap dalam bentuk subset yang diaudit.
- Tidak boleh menampilkan angka hari menuju matang sebelum ada data waktu yang valid.

## Review Focus

- Gambar rusak atau bukan gambar harus menghasilkan error API yang terbaca, bukan crash.
  Test: `test_invalid_image_returns_422` di task 2.
- Tidak ada model custom harus tetap membuat aplikasi dapat dibuka dengan status model yang jelas, bukan mengarang hasil.
  Test: `test_no_model_reports_unavailable` di task 2.
- Confidence rendah dan objek di luar lima kelas harus ditolak.
  Test: `test_low_confidence_is_rejected` di task 2.
- Beberapa bounding box yang berdekatan harus dipertahankan sebagai objek terpisah.
  Test: `test_normalize_keeps_multiple_objects` di task 1.
- Kamera ditolak browser atau backend mati harus memiliki pesan tindakan yang jelas.
  Test: frontend smoke test di task 4.

---

### Task 1: Backend domain schema dan inference provider

**Files:**
- Create: `backend/app/schemas/prediction.py`
- Create: `backend/app/inference/base.py`
- Create: `backend/app/inference/demo_provider.py`
- Create: `backend/app/inference/ultralytics_provider.py`
- Create: `backend/app/services/prediction_service.py`
- Create: `backend/tests/test_prediction_domain.py`
- Create: `backend/pyproject.toml`

**Interfaces:**
- `PredictionService.predict(image_bytes: bytes) -> PredictionResponse`
- `InferenceProvider.predict(image: PIL.Image.Image) -> list[RawDetection]`
- `RawDetection` carries `label`, `confidence`, `box`, and optional `stage`.
- Response labels use Indonesian display names while retaining stable English `class_name` values.

- [ ] Tulis test schema untuk response kosong, multi-object, bounding box bounds, dan tahap kematangan yang belum tersedia.
- [ ] Jalankan `py -m pytest backend/tests/test_prediction_domain.py -q` dan pastikan test gagal sebelum implementasi.
- [ ] Implementasikan Pydantic models, provider protocol, normalizer, dan aturan confidence threshold default `0.45`.
- [ ] Implementasikan provider demo yang hanya dipakai untuk test UI dan selalu diberi `mode=demo`.
- [ ] Implementasikan provider Ultralytics yang memuat bobot sekali, mendukung custom class prompt/model names, dan mengembalikan objek terpisah.
- [ ] Jalankan pytest sampai PASS.

### Task 2: FastAPI API, model status, dan audit manifest

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/main.py`
- Create: `backend/app/api/routes.py`
- Create: `backend/app/services/model_registry.py`
- Create: `backend/tests/test_api.py`
- Create: `research/dataset_registry.yaml`
- Create: `research/audit_dataset.py`
- Create: `research/README.md`
- Create: `backend/.env.example`
- Modify: `backend/app/services/prediction_service.py`

**Interfaces:**
- `GET /health` returns `{status, model_loaded, mode}`.
- `GET /api/model-info` returns model name, classes, stage availability, and limitation text.
- `POST /api/predict/image` accepts multipart `file` and returns `PredictionResponse`.
- `GET /api/calibration-status` always reports day estimate unavailable in MVP.
- `python research/audit_dataset.py --manifest research/dataset_registry.yaml` writes an audit JSON without downloading archives.

- [ ] Tulis API tests untuk health, model info, invalid image, no-model status, and prediction response.
- [ ] Jalankan test untuk memastikan gagal sebelum route/service tersedia.
- [ ] Implementasikan settings berbasis environment, CORS localhost, size limit 10 MB, dan error response yang aman.
- [ ] Implementasikan model registry yang memilih custom weights jika ada, lalu fallback ke provider yang terkonfigurasi tanpa menyembunyikan status unavailable.
- [ ] Implementasikan registry dataset dengan URL, lisensi, kelas, status audit, dan catatan provenance.
- [ ] Implementasikan audit format/metadata lokal yang tidak menghapus data dan tidak mengunduh otomatis.
- [ ] Jalankan `py -m pytest backend/tests -q` dan `py research/audit_dataset.py --manifest research/dataset_registry.yaml`.

### Task 3: React camera/upload interface

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/api.ts`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/components/CameraPanel.tsx`
- Create: `frontend/src/components/ResultPanel.tsx`
- Create: `frontend/src/components/DetectionCanvas.tsx`
- Create: `frontend/src/components/StatusBanner.tsx`

**Interfaces:**
- `POST /api/predict/image` is called with `FormData` and an abort timeout of 8 seconds.
- `CameraPanel` exposes snapshot, upload, and camera error state.
- `ResultPanel` renders every object by stable `id`, confidence, stage, and limitations.
- No frontend copy may imply disease diagnosis or day estimate availability.

- [ ] Tulis type contracts and a pure canvas coordinate test for normalized boxes.
- [ ] Implementasikan camera permission flow, file upload, snapshot conversion, and loading/error states.
- [ ] Implementasikan canvas overlay that preserves aspect ratio and draws every detection separately.
- [ ] Implementasikan result list, model status, empty state, and offline limitation notice.
- [ ] Implementasikan responsive visual system with dark leaf base, yellow/ripe accents, visible focus, and reduced-motion support.
- [ ] Jalankan `npm install`, `npm run build`, dan `npm run lint` bila script tersedia.

### Task 4: Local run scripts and web smoke test

**Files:**
- Create: `run_backend.cmd`
- Create: `run_frontend.cmd`
- Create: `run_all.cmd`
- Create: `README.md`
- Create: `frontend/tests/smoke.py`
- Create: `backend/requirements.txt`

**Interfaces:**
- `run_all.cmd` starts backend on `127.0.0.1:8000` and frontend on `127.0.0.1:5173`.
- README documents Python setup, Node setup, optional model weights, offline use, and known limitations.
- Smoke test verifies page load, upload control, camera unavailable message path, and API error display.

- [ ] Tulis README setup from a clean Windows shell and include exact commands.
- [ ] Add dependency pins compatible with Python 3.11 and the CPU laptop.
- [ ] Add command wrappers without embedding credentials or forcing downloads.
- [ ] Run `py -m pip install -r backend/requirements.txt` in the project environment and start both servers.
- [ ] Run `py frontend/tests/smoke.py --help` first, then execute the smoke test against the running app.
- [ ] Verify backend `/docs`, frontend page, upload flow, and clear unavailable-model behavior.

### Task 5: Dataset subset preparation and training handoff

**Files:**
- Create: `research/download_subset.py`
- Create: `research/prepare_yolo.py`
- Create: `research/train_detector.py`
- Create: `research/evaluate.py`
- Create: `research/README.md` updates
- Create: `data/manifests/README.md`

**Interfaces:**
- Scripts require explicit `--source` and `--output`, never download by default.
- Preparation preserves original labels and writes normalized YOLO labels plus a manifest.
- Training accepts `--data`, `--model`, `--epochs`, `--imgsz`, and `--device cpu`.
- Evaluation writes mAP, per-class recall, crowded subset counts when available, latency, and limitation notes.

- [ ] Implementasikan dry-run subset preparation and manifest validation first.
- [ ] Add a small audited source configuration for five target classes, without claiming Indonesian origin unless metadata proves it.
- [ ] Implementasikan training command that can produce `models/fruit_detector.pt` when the user supplies/downloads compatible annotations.
- [ ] Implementasikan evaluation report and model metadata consumed by the API.
- [ ] Run dry-run audit and training help checks. Do not start a multi-gigabyte download automatically.

## Verification Checklist

- `py -m pytest backend/tests -q`
- `py research/audit_dataset.py --manifest research/dataset_registry.yaml`
- `npm run build` in `frontend`
- `py frontend/tests/smoke.py --help`, then smoke test with both servers
- Open `http://127.0.0.1:5173` and verify upload, camera permission handling, loading, error, and no-model states.
- Open `http://127.0.0.1:8000/docs` and verify the API schema.
- Confirm no credentials are written and no dataset is downloaded without an explicit command.

## Known Gaps

- The MVP cannot honestly estimate days to maturity without longitudinal time-labelled data.
- Disease/defect explanation is deliberately disabled in MVP.
- Public internet test images do not prove performance on the user's own camera.
- If no compatible custom weights are supplied, the app remains usable as a UI/API demo but reports model unavailable instead of inventing predictions.
