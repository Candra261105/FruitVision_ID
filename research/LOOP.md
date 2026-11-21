# Loop otomatis FruitVision

`research/fruitvision_loop.py` menjalankan siklus latih → evaluasi → gate → ulangi
untuk semua target, tanpa menunggu interaksi manusia. Setiap langkah dicatat di
`runs/loop/iterations.jsonl` (satu baris JSON per percobaan) dan `runs/loop/loop.log`.

## Target

| Target | Yang dijalankan | Syarat lulus |
|---|---|---|
| `maturity` | klasifikasi tomat (3 kelas) | gate advanced: macro-F1 ≥ 0,75, balanced ≥ 0,75, support ≥ 20, tiap kelas ≥ 5 |
| `rot` | klasifikasi tomat (2 kelas) | sama |
| `grade` | klasifikasi tomat A/B/C | sama (model-metrik 0,70 di loop) |
| `disease` | klasifikasi mentimun (3 kelas buah) | sama |
| `detector` | YOLO11n 3 kelas | mAP50 ≥ 0,55 dan tiap kelas ≥ 0,40 |
| `integration` | pytest backend, build frontend, status model | semua exit 0 dan minimal 1 model eligible |

Target `detector` sengaja tidak menurunkan ambang saat gagal. Jika bobot terbaik
masih di bawah ambang, model tidak dipasang dan aplikasi tetap memakai YOLO-World
sebagai fallback dengan status apa adanya.

## Perintah

```powershell
# semua target
& .\.venv\Scripts\python.exe -m research.fruitvision_loop

# satu target
& .\.venv\Scripts\python.exe -m research.fruitvision_loop --only disease

# laporan
& .\.venv\Scripts\python.exe research\loop_report.py
```

## Berhenti dan lanjut

- `STOP_LOOP` di root proyek: loop berhenti sebelum target berikutnya.
- `STOP_TRAINING` di root proyek: training berhenti di akhir epoch berjalan,
  checkpoint `last.pt` tetap disimpan sehingga bisa dilanjutkan.
- Training yang terputus dilanjutkan otomatis: `advanced_train_loop` memakai
  `resume=True` bila checkpoint valid ditemukan (`find_valid_checkpoint`).

## Dataset

- Manifest dan audit: `data/advanced/*.jsonl` + `*.audit.json`.
- Split dikunci di manifest (train/validation/test) memakai pengelompokan
  average-hash, jadi tidak ada gambar mirip yang bocor antar-split.
- Augmentasi hanya diterapkan Ultralytics pada split train; test holdout tidak
  diaugmentasi.

## Aturan kejujuran

- Gate ditulis oleh `research/evaluate_advanced_classifier.py`, bukan manual.
- Label evaluasi harus subset dari `research/advanced_taxonomy.yaml` (minimal 2 kelas).
- Model tanpa gate `eligible` tidak dimuat backend, dan hasil `review` tidak
  diterjemahkan menjadi diagnosis biologis.
