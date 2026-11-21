# Advanced Condition Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menyiapkan dan menjalankan pipeline terpisah untuk mendeteksi penyakit, kematangan, busuk, dan grade pada `cucumber`, `tomato`, dan `green beans`, dengan model hanya dapat dipakai setelah lolos gate evaluasi.

**Architecture:** Pipeline memakai dataset object-level/crop dengan manifest tervalidasi, split berdasarkan session/source, classifier terpisah per task, runner yang resumable, evaluator dengan calibration dan confusion matrix, lalu sidecar gate yang dibaca registry model. Dataset atau task tanpa label yang sesuai berhenti pada status `blocked_no_dataset`, bukan membuat prediksi palsu.

**Tech Stack:** Python 3.11, PyTorch/Ultralytics yang sudah dipakai proyek, Pillow/OpenCV, JSONL/YAML, pytest, FastAPI/React TypeScript untuk integrasi.

**Spec:** `docs/superpowers/specs/2026-11-18-advanced-condition-training-design.md`

## Global Constraints

- Dataset detector jenis buah tidak boleh dipakai sebagai label penyakit, kematangan, busuk, atau grade.
- Model task baru tidak boleh dipromosikan tanpa provenance, split audit, evaluasi test tetap, dan sidecar `status: "eligible"`.
- `unknown` menjadi status abstain/review, bukan kelas positif yang dipaksakan.
- Split harus berdasarkan session/source/batch/object lineage, bukan frame acak.
- Training dapat dihentikan dengan `STOP_TRAINING` dan dilanjutkan dari checkpoint valid.
- Model aplikasi tidak diubah otomatis oleh runner.
- Training detector `runs/fruitvision-loop` yang sedang berjalan tidak dihentikan atau dimulai ulang.
- API key dan foto pengguna tidak dikirim ke provider eksternal oleh pipeline dataset.

## Review Focus

- Manifest dengan bbox/label/object_id salah harus ditolak sebelum training. Test: validator menolak bbox di luar rentang, file hilang, dan taxonomy task yang salah pada Task 1.
- Foto dari session yang sama tidak boleh tersebar ke train dan test. Test: splitter menghasilkan satu split untuk seluruh session pada Task 1.
- Checkpoint korup atau status JSON rusak tidak boleh membuat run dianggap selesai. Test: runner mengabaikan checkpoint korup dan memulihkan status pada Task 2.
- Dataset yang belum ada tidak boleh memulai training kosong. Test: runner menulis `blocked_no_dataset` tanpa membuat bobot pada Task 2.
- Model dengan metrik buruk atau provenance tidak lengkap tidak boleh memiliki gate `eligible`. Test: gate menulis `rejected` pada Task 3.

---

### Task 1: Dataset manifest, taxonomy, dan provenance validator

**Files:**
- Create: `research/advanced_taxonomy.yaml`
- Create: `research/advanced_manifest_schema.py`
- Create: `research/validate_advanced_dataset.py`
- Create: `research/split_advanced_dataset.py`
- Create: `backend/tests/test_advanced_dataset.py`
- Create: `data/advanced/README.md`

**Interfaces:**
- `load_taxonomy(path: Path) -> dict[str, list[str]]`
- `validate_manifest(manifest_path: Path, image_root: Path, taxonomy_path: Path) -> ValidationReport`
- `split_manifest(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]`
- `ValidationReport.valid: bool`, `.errors: list[str]`, `.counts: dict[str, int]`

- [ ] Write failing tests for valid records, missing images, invalid bbox, unsupported task/fruit, missing provenance, duplicate exact hashes, and session-level split isolation.
- [ ] Run `pytest backend/tests/test_advanced_dataset.py -q` and confirm the new tests fail.
- [ ] Implement taxonomy with task labels: disease, maturity, rot, grade. Keep disease taxonomy configurable because no disease list is yet verified for the target plants.
- [ ] Implement manifest validation for image existence, normalized bbox, object_id, fruit class, task label, source, permission/license, session_id, and label_version.
- [ ] Implement exact hash and perceptual-hash duplicate checks where image decoding is available. Report duplicates without deleting source files.
- [ ] Implement deterministic group split by `session_id`, source, and object lineage. Ignore any user-provided `split` field when generating the final split.
- [ ] Add `data/advanced/README.md` with the JSONL record format and the exact command for validation/splitting.
- [ ] Run the focused tests and confirm PASS.

### Task 2: Advanced training runner and blocked-task handling

**Files:**
- Create: `research/advanced_training_matrix.yaml`
- Create: `research/advanced_train_loop.py`
- Create: `research/advanced_training_status.py`
- Create: `backend/tests/test_advanced_train_loop.py`
- Modify: `README.md` in the training section

**Interfaces:**
- `load_advanced_matrix(path: Path) -> list[dict[str, Any]]`
- `run_advanced_task(task: dict[str, Any], manifest: Path, output: Path, device: str) -> dict[str, Any]`
- `write_status_atomic(path: Path, payload: dict[str, Any]) -> None`
- `find_valid_checkpoint(run_dir: Path) -> Path | None`

- [ ] Write failing tests for absent manifest, no valid records for a task, corrupt checkpoint, atomic status, stop-file handling, and completed-run resume.
- [ ] Run `pytest backend/tests/test_advanced_train_loop.py -q` and confirm failure.
- [ ] Implement one independent run directory per task: `runs/advanced-condition/<task>/`.
- [ ] Implement statuses `blocked_no_dataset`, `running`, `interrupted`, `failed`, and `completed`.
- [ ] Implement classifier training on object crops only after the validator reports enough labeled records. Use the existing local PyTorch/Ultralytics stack and do not download weights implicitly.
- [ ] Implement `STOP_TRAINING`, valid checkpoint loading, atomic JSON writes, manifest hash, taxonomy version, and config capture.
- [ ] Add four matrix entries but ensure missing task data is skipped with `blocked_no_dataset`.
- [ ] Run focused tests and a dry run against the current local data. Expected result: all four advanced tasks are blocked, and the detector loop remains untouched.

### Task 3: Evaluation, calibration, and model gate

**Files:**
- Create: `research/evaluate_advanced.py`
- Create: `research/gate_advanced_model.py`
- Create: `backend/tests/test_advanced_evaluation.py`

**Interfaces:**
- `evaluate_task(model_path: Path, manifest: Path, task: str, output: Path) -> dict[str, Any]`
- `write_advanced_gate(model_path: Path, evaluation: dict[str, Any], provenance: dict[str, Any]) -> Path`

- [ ] Write failing tests for macro-F1/balanced accuracy aggregation, ordinal error for maturity/grade, validation-only calibration, insufficient samples, missing provenance, and rejected metrics.
- [ ] Run focused tests and confirm failure.
- [ ] Implement fixed test evaluation with per-class and per-fruit metrics, confusion matrix, sample counts, and confidence intervals where sample size permits.
- [ ] Implement calibration on validation only and store calibration method/threshold/abstain policy in the gate sidecar.
- [ ] Implement ordinal error metrics for maturity and grade without replacing macro-F1.
- [ ] Implement gate statuses `eligible`, `rejected`, and `blocked` with explicit reasons. Do not guess production thresholds before real metrics exist. Use conservative configured minimums and report when they are not met.
- [ ] Run focused tests and verify no rejected model receives `eligible`.

### Task 4: Registry and backend capability integration

**Files:**
- Create: `backend/app/services/advanced_model_registry.py`
- Modify: `backend/app/services/model_registry.py`
- Modify: `backend/app/services/prediction_service.py`
- Modify: `backend/app/schemas/prediction.py`
- Create: `backend/tests/test_advanced_model_registry.py`

**Interfaces:**
- `load_advanced_models(model_root: Path) -> dict[str, AdvancedModelRuntime]`
- `AdvancedModelRuntime.status: Literal["eligible", "unavailable", "rejected", "blocked"]`
- `AdvancedModelRuntime.predict(crop: Image.Image) -> AdvancedPrediction | None`

- [ ] Write failing tests proving a missing gate, rejected gate, or wrong task taxonomy stays unavailable, while an eligible model loads only for its own task.
- [ ] Run focused tests and confirm failure.
- [ ] Implement registry loading only sidecars with `status: eligible`, matching task, class taxonomy, manifest hash, and model file.
- [ ] Extend capability responses so each feature can independently move from `Belum tersedia` to an explicitly qualified available/review state.
- [ ] Keep detector output unchanged when advanced models are unavailable or rejected.
- [ ] Run focused tests and confirm default current application still reports all four features unavailable.

### Task 5: Frontend status, review, and final verification

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/components/ResultPanel.tsx`
- Modify: `frontend/src/lib/capabilities.ts`
- Create/modify: `frontend/src/lib/advancedCapabilities.test.ts`
- Modify: `README.md`

**Interfaces:**
- Capability UI must display task name, status, confidence/review state, model version, and limitation.
- Unavailable/rejected/blocked must remain distinct from a negative biological result.

- [ ] Write failing frontend tests for four unavailable states, one eligible state, one rejected state, and review/abstain output.
- [ ] Run `cmd /c npm test --prefix frontend -- --run` and confirm failure.
- [ ] Implement independent UI rendering per task. Do not imply `healthy`, `not rotten`, or a specific grade when the task is unavailable or abstained.
- [ ] Add run commands, dataset requirements, and current blocked status to `README.md`.
- [ ] Run full verification: backend tests, frontend tests, TypeScript/build, dataset dry run, and status inspection.
- [ ] Report which tasks are truly trained and eligible. If no valid target dataset was acquired, report all four as `blocked_no_dataset` and leave the UI unchanged at `Belum tersedia`.

## Execution Order

Run Tasks 1 and 2 in sequence, then Task 3, then Task 4, and finally Task 5. Dataset acquisition is a prerequisite to a real training run. Public datasets may be added only after license and target-class audit; metadata-only candidates must not enter training.

## Completion Criteria

The project is complete only when the runner, validator, evaluator, gate, registry, and UI tests pass, and each feature has an explicit evidence-backed status. A scaffold with no target labels is considered technically complete but remains operationally `blocked_no_dataset`; it must not be reported as a trained or accurate model.
