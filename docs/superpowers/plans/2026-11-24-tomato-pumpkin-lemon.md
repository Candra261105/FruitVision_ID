# Tomato Pumpkin Lemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membuat dataset eksperimen baru dan integrasi aplikasi untuk detector tiga kelas `0 tomato`, `1 pumpkin`, `2 lemon`, dengan audit label, komposit sintetis, crop per objek, pemilihan objek, peringatan kualitas, dan indikator warna HSV yang tidak di-overclaim sebagai diagnosis.

**Architecture:** Dataset lama tetap utuh. Pipeline baru memindai sumber existing, memvalidasi provenance dan label, membuat split berbasis source group, lalu menghasilkan komposit hanya dari sumber train. Backend memakai taxonomy baru secara terpusat, membuat crop dan indikator kualitas per detection, sedangkan frontend menampilkan daftar crop yang dapat dipilih dan rincian objek terpilih. Model baru tidak menggantikan model web sampai gate evaluasi lulus.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics YOLO, Pillow/OpenCV yang sudah dipakai project, FastAPI/Pydantic, React/TypeScript/Vite, pytest, npm build.

**Spec:** `docs/superpowers/specs/2026-11-24-tomato-pumpkin-lemon-design.md`

## Global Constraints

- Dataset lama tidak ditimpa.
- Urutan kelas wajib `0 tomato`, `1 pumpkin`, `2 lemon`.
- `cucumber` dan `green beans` tidak boleh direlabel menjadi `pumpkin` atau `lemon`.
- Data sintetis hanya boleh masuk `train`.
- `val`, `test`, dan `external_holdout` harus berasal dari gambar asli yang tidak menjadi sumber komposit train.
- Pemisahan harus berbasis `source_id` atau grup sumber, exact hash, dan perceptual hash untuk mencegah leakage.
- HSV hanya disebut indikator warna sampai ada ground truth dan gate kondisi yang tervalidasi.
- Model baru tidak otomatis mengganti model web.
- Jika sumber valid pumpkin atau lemon kurang, dataset tetap dilaporkan sebagai eksperimen parsial dan gate model menandainya belum layak deployment.
- Semua perintah dan hasil yang belum dijalankan harus ditandai belum diuji.

## Review Focus

- Sumber pumpkin/lemon tidak tersedia atau hanya muncul di laporan metadata: audit harus menghasilkan laporan parsial, bukan label tebakan. Test: audit menandai kelas tanpa sumber valid.
- Crop dari `val`/`test` masuk komposit train: split manifest harus menolak source group lintas split. Test: source ID yang sama di dua split gagal validasi.
- Komposit mengandung objek target lain tanpa label: background harus disaring atau ditolak. Test: komposit dengan background target tak berlabel ditolak.
- Objek overlap atau terpotong: box harus mengikuti kebijakan visible/full-object dan memberi warning. Test: overlap tinggi dan box menyentuh tepi menghasilkan warning stabil.
- UI menampilkan HSV sebagai kematangan tervalidasi: schema dan copy harus menyebut indikator warna serta status review. Test: hasil tanpa model kondisi tidak menampilkan klaim diagnosis.

---

### Task 1: Audit sumber gambar dan taxonomy baru

**Files:**
- Create: `research/tomato_pumpkin_lemon_audit.py`
- Create: `backend/tests/test_tomato_pumpkin_lemon_audit.py`
- Create: `data/processed/tomato-pumpkin-lemon/README.md`
- Create: `data/processed/tomato-pumpkin-lemon/dataset_report.json`
- Create: `data/processed/tomato-pumpkin-lemon/source_manifest.json`

**Interfaces:**
- Produces `audit_sources(root: Path, output: Path) -> AuditReport`.
- Produces deterministic class IDs: `tomato=0`, `pumpkin=1`, `lemon=2`.
- Produces source records containing path, source ID, image hash, perceptual hash when available, label path, license/provenance, class counts, and validation errors.

- [ ] **Step 1: Write failing tests** for deterministic class order, missing label detection, invalid YOLO box detection, and partial-class reporting when pumpkin/lemon sources are absent.
- [ ] **Step 2: Run** `pytest backend/tests/test_tomato_pumpkin_lemon_audit.py -q` and confirm failure because the audit module is not implemented.
- [ ] **Step 3: Implement** the scanner over existing `data`, `research`, and documented source locations. Do not reinterpret old class names. Use exact hash and a deterministic perceptual-hash fallback only for duplicate grouping.
- [ ] **Step 4: Run** the focused tests and verify the report records every missing/invalid source without creating labels.
- [ ] **Step 5: Run** the audit against the project root and inspect class counts, duplicate groups, licenses, and missing classes.
- [ ] **Step 6: Commit** `feat(data): audit tomato pumpkin lemon sources`.

### Task 2: Build verified three-class dataset and source-based splits

**Files:**
- Create: `research/build_tomato_pumpkin_lemon_dataset.py`
- Create: `backend/tests/test_tomato_pumpkin_lemon_dataset.py`
- Create: `data/processed/tomato-pumpkin-lemon/data.yaml`
- Create: `data/processed/tomato-pumpkin-lemon/data-external.yaml`
- Create: `data/processed/tomato-pumpkin-lemon/images/{train,val,test,external_holdout}/`
- Create: `data/processed/tomato-pumpkin-lemon/labels/{train,val,test,external_holdout}/`
- Create: `data/processed/tomato-pumpkin-lemon/split_manifest.json`

**Interfaces:**
- `build_dataset(audit: Path, output: Path, seed: int = 20261124) -> DatasetReport`.
- `validate_dataset(root: Path, allow_partial: bool = True) -> DatasetReport`.
- `source_group_split(records: list[SourceRecord], seed: int) -> SplitManifest`.

- [ ] **Step 1: Write failing tests** for class IDs, source-group split isolation, exact/perceptual duplicate isolation, relative YAML paths, and valid/invalid YOLO labels.
- [ ] **Step 2: Run** the focused test file and confirm expected failures.
- [ ] **Step 3: Implement** copying/linking of verified original images and labels only. Re-map only verified classes, preserve original source provenance, and keep an `external_holdout` split when valid external data exists.
- [ ] **Step 4: Implement** split validation that rejects the same source group, exact hash, or near-duplicate across train/val/test/external holdout.
- [ ] **Step 5: Generate** the dataset from the audit result. If pumpkin or lemon has zero valid sources, keep the class in `data.yaml` but mark it `unavailable_for_training` in the report and do not invent labels.
- [ ] **Step 6: Run** dataset validation and inspect per-class image/object counts and split hashes.
- [ ] **Step 7: Commit** `feat(data): build verified tomato pumpkin lemon dataset`.

### Task 3: Generate conservative synthetic composites

**Files:**
- Create: `research/generate_tomato_pumpkin_lemon_composites.py`
- Create: `backend/tests/test_tomato_pumpkin_lemon_composites.py`
- Modify: `data/processed/tomato-pumpkin-lemon/dataset_report.json`
- Create: `data/processed/tomato-pumpkin-lemon/synthetic_manifest.json`

**Interfaces:**
- `generate_composites(dataset_root: Path, count: int, seed: int, max_synthetic_ratio: float = 1.0) -> SyntheticReport`.
- `validate_composite(image: Path, label: Path, manifest: CompositeRecord) -> None`.

- [ ] **Step 1: Write failing tests** for deterministic seed output, box generation from pasted objects, overlap metadata, edge metadata, and rejection of source records from val/test/external holdout.
- [ ] **Step 2: Run** the focused tests and confirm failure.
- [ ] **Step 3: Implement** compositing only from train-source objects and approved backgrounds. Record source IDs, class IDs, transforms, positions, visible/full-object policy, seed, and output hashes.
- [ ] **Step 4: Add** controlled cases for separated, adjacent, touching, overlapping, small, dark, bright, blurred, and noisy objects. Limit synthetic counts per class and report the real-to-synthetic ratio.
- [ ] **Step 5: Reject** backgrounds containing unlabeled target objects when detected by available annotations or audit rules. Do not claim perfect background screening where it is not possible.
- [ ] **Step 6: Run** manifest/label validation and verify all synthetic files are under `images/train` with matching labels.
- [ ] **Step 7: Commit** `feat(data): add audited synthetic composite augmentation`.

### Task 4: Add training, evaluation, and model gate for the new dataset

**Files:**
- Create: `runs/loop/train_tomato_pumpkin_lemon.py`
- Create: `training_presets/tomato_pumpkin_lemon_{50,60,70,80,90,100}_epochs.cmd`
- Create: `research/evaluate_tomato_pumpkin_lemon.py`
- Create: `backend/tests/test_tomato_pumpkin_lemon_training.py`
- Create: `research/results/tomato-pumpkin-lemon/README.md`
- Create: `research/results/tomato-pumpkin-lemon/gate.json`

**Interfaces:**
- CLI: `python runs/loop/train_tomato_pumpkin_lemon.py --epochs N [--weights PATH] [--name NAME]`.
- `evaluate_model(weights: Path, dataset_yaml: Path, external_yaml: Path | None) -> EvaluationReport`.
- Gate sidecar records taxonomy, weights hash, dataset manifest hash, split hash, per-class metrics, support, and status.

- [ ] **Step 1: Write failing tests** for epoch validation, class-order validation, refusal to deploy a partial class set, and gate status when one class lacks minimum support.
- [ ] **Step 2: Run** focused tests and confirm failure.
- [ ] **Step 3: Implement** the runner using the existing Ultralytics configuration style, with explicit `--epochs`, deterministic seed, output folder, and no automatic web deployment.
- [ ] **Step 4: Implement** evaluation on real-only val/test/external holdout. Use val for checkpoint/threshold choice and reserve test for the final report. Report precision, recall, mAP50, mAP50-95, confusion matrix, and per-class support.
- [ ] **Step 5: Implement** a gate that remains `rejected` or `partial` if any required class lacks valid source/support or fails the configured minimum. Do not choose thresholds from test.
- [ ] **Step 6: Run** `--help`, compile checks, and a short dry validation without starting a long training job. Actual 50–100 epoch training remains a user-triggered operation.
- [ ] **Step 7: Commit** `feat(training): add tomato pumpkin lemon training and evaluation`.

### Task 5: Replace detector taxonomy centrally in the backend

**Files:**
- Modify: `backend/app/inference/ultralytics_provider.py`
- Modify: `backend/app/services/model_registry.py`
- Modify: `backend/app/services/prediction_service.py`
- Modify: `backend/app/schemas/prediction.py`
- Modify: relevant backend tests under `backend/tests/`

**Interfaces:**
- Central taxonomy constant: `DETECTOR_CLASSES = ("tomato", "pumpkin", "lemon")`.
- `Detection` gains optional `crop_image`, `color_indicator`, and stable quality metadata only if schema compatibility is preserved.
- HSV result shape: `ColorIndicator(hue_segments: dict[str, float], saturation: float, value: float, status: Literal["available","review"])`.

- [ ] **Step 1: Write failing tests** for new class normalization, crop generation, HSV percentages summing to 100 within tolerance, edge/small/overlap warnings, and partial-model status.
- [ ] **Step 2: Run** relevant backend tests and confirm failures against old taxonomy/schema.
- [ ] **Step 3: Update** taxonomy and model metadata centrally. Existing old-model tests must be separated or explicitly marked as legacy behavior rather than silently passing old classes through the new detector.
- [ ] **Step 4: Implement** per-detection crop from normalized boxes with bounded padding, base64/data URL output, and deterministic detection IDs.
- [ ] **Step 5: Implement** HSV as a color indicator with conservative masks and a review status. Do not convert raw HSV into validated maturity/rot labels without a gated condition model.
- [ ] **Step 6: Implement** warning rules for touching edge, small area, low confidence, close centers, and high IoU overlap. Keep warnings in the API response.
- [ ] **Step 7: Run** all backend tests and verify legacy behavior is intentionally handled.
- [ ] **Step 8: Commit** `feat(api): support tomato pumpkin lemon per-object results`.

### Task 6: Add crop selection and per-object result UI

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/ResultPanel.tsx`
- Create: `frontend/src/components/ObjectCropSelector.tsx`
- Create: `frontend/src/components/ColorIndicatorChart.tsx`
- Modify: `frontend/src/styles.css` or the existing frontend stylesheet
- Create/modify: frontend tests if the project test setup supports them

**Interfaces:**
- `ObjectCropSelector` receives `detections: Detection[]`, `selectedId: number | null`, and `onSelect(id: number): void`.
- `ColorIndicatorChart` receives normalized percentages and renders a pie/donut chart with text fallback for accessibility.
- Selected object is identified by stable `detection.id`, not array position.

- [ ] **Step 1: Write failing UI tests or pure-component tests** for crop list rendering, selection state, selected-object detail, warning display, and pie percentage labels.
- [ ] **Step 2: Run** the focused frontend test command and confirm failure or, if no test runner exists, add a minimal testable pure helper and run TypeScript checks.
- [ ] **Step 3: Implement** the crop selector with keyboard-accessible buttons, thumbnail/crop image, class, confidence, and warning markers.
- [ ] **Step 4: Implement** the selected-object panel with `Gambar sudah diperjelas ✓`, `buah sudah terdeteksi ✓`, object confidence, HSV color indicator, result status, and explicit manual-verification text.
- [ ] **Step 5: Render** the chart only from normalized indicator data. Use labels such as `warna hijau`, `warna kuning`, `warna merah`, or configured HSV bands unless a validated maturity model exists.
- [ ] **Step 6: Add** clear per-object warnings: `Objek menyentuh tepi gambar`, `Objek berukuran kecil`, `Objek saling berdekatan`, and `Perlu verifikasi lanjutan oleh pengguna.`
- [ ] **Step 7: Run** TypeScript/build and browser interaction checks for upload, detection, crop selection, and no-detection states.
- [ ] **Step 8: Commit** `feat(frontend): add selectable crops and per-object results`.

### Task 7: Documentation, integration verification, and deployment gate

**Files:**
- Modify: `README.md`
- Modify: `data/README.md`
- Create: `docs/tomato-pumpkin-lemon-training.md`
- Modify: `.gitignore` if needed

**Interfaces:**
- Documentation names the exact dataset path, class IDs, commands, provenance limits, synthetic ratio, evaluation protocol, and manual activation procedure.

- [ ] **Step 1: Write documentation checks** for exact class order, dataset path, training commands, and warning that epoch count does not guarantee accuracy.
- [ ] **Step 2: Update** README and dataset README with clone/setup, dataset generation, preset training, evaluation, and gate review instructions.
- [ ] **Step 3: Verify** no dataset bulk images, secrets, cache, or unapproved model artifacts enter Git.
- [ ] **Step 4: Run** the full backend test suite, frontend build, dataset validation, and API smoke test.
- [ ] **Step 5: Run** browser checks: upload, enhanced-image status, detection result, crop selection, color chart, warnings, and AI fallback.
- [ ] **Step 6: Confirm** the web application still uses the existing model unless the new model gate explicitly says eligible and a manual activation step is performed.
- [ ] **Step 7: Commit** `docs(fruitvision): document tomato pumpkin lemon pipeline`.
- [ ] **Step 8: Push** only after the working tree is clean and verification output is recorded.

## Verification Commands

```powershell
cd C:\Users\dragon\Downloads\FruitVision_ID
& .\.venv\Scripts\python.exe -m pytest -q
cmd /c npm run build --prefix frontend
& .\.venv\Scripts\python.exe research\build_tomato_pumpkin_lemon_dataset.py --help
& .\.venv\Scripts\python.exe runs\loop\train_tomato_pumpkin_lemon.py --help
```

Long training commands are intentionally not run as part of implementation
verification. They are started manually after the dataset report and gate
inputs have been reviewed.
