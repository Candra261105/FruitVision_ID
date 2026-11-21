# Progress advanced condition training

Plan: `docs/superpowers/plans/2026-11-18-advanced-condition-training.md`

## Rulings

- Ruling: Dataset publik tidak diunduh otomatis — pencarian web sesi ini gagal dan lisensi/cakupan target belum dapat diverifikasi, sehingga memakai sumber tanpa audit akan berisiko membuat model tidak sah atau salah klaim.
- Ruling: `advanced_taxonomy.yaml` menyediakan label awal `healthy/unknown`, tahap kematangan, busuk, dan grade — daftar penyakit nyata tetap harus diperbarui setelah sumber data/validator ditentukan.
- Ruling: Semua task yang tidak memiliki manifest valid ditulis sebagai `blocked_no_dataset` — ini menjaga aplikasi tidak mengklaim model yang belum pernah dilatih.

## Task 1: complete

Validator manifest, exact duplicate check, taxonomy, dan session-level split selesai.
Test: `backend/tests/test_advanced_dataset.py` -> 6 passed.

## Task 2: complete

Runner empat task, status atomik, valid checkpoint, stop-file guard, dan dry run selesai.
Test: `backend/tests/test_advanced_train_loop.py` -> 4 passed.
Dry run: `disease`, `maturity`, `rot`, `grade` -> `blocked_no_dataset`, tanpa bobot.

## Task 3: complete

Evaluator metrik klasifikasi/ordinal dan gate `eligible/rejected` selesai.
Test: `backend/tests/test_advanced_evaluation.py` -> 3 passed.

## Task 4: complete

Registry advanced model dan capability backend independen selesai.
Test: `backend/tests/test_advanced_model_registry.py` + `test_prediction_domain.py` -> 9 passed.

## Task 5: complete

UI regression test, README, dan perintah dry run selesai.
Test: frontend -> 4 passed.
