# Laporan Pipeline A FruitVision ID

Tanggal: 18 November 2026

## Data dan audit

Sumber object-detection dengan lisensi tercatat:

| Sumber | Format | Gambar lokal | Lisensi |
|---|---:|---:|---|
| LVIS Fruits and Vegetables mirror GitHub | YOLO | 180 | MIT |
| Bangladesh Smartphone Detection | Parquet | 3.534 | CC-BY-4.0 |
| Fruit Detection Worldwide | Parquet | 565 | CC-BY-4.0 |
| Kitchen Fruit Vegetable Detection | COCO | 120 | CC-BY-4.0 |

Audit mendeteksi 0 box rusak pada laporan sumber yang selesai dan 0 gambar unresolved pada Kitchen. Hash SHA-256 dipakai untuk deduplikasi lintas sumber dan split.

## Top 3

Top 3 dihitung dari box valid gabungan setelah normalisasi kelas dan deduplikasi gambar:

| Kelas | Box valid | Gambar unik | Sumber |
|---|---:|---:|---:|
| cucumber | 12.906 | 1.484 | 2 |
| tomato | 11.907 | 2.145 | 3 |
| green beans | 7.468 | 1.710 | 1 |

`Tomato`, `tomatoes`, `tomat`, dan `tomato` dinormalisasi menjadi `tomato`. `banana` dan `green banana` tetap terpisah. `orange`, `lemon`, `lime`, dan `mandarin orange` tidak digabung.

## Dataset training

Kitchen sengaja dipisahkan sebagai test lintas-domain. Dataset training gabungan berada di `data/processed/combined-top3-final`:

- 3.191 gambar unik
- train: 2.552
- validation: 319
- test: 320
- kelas: `cucumber`, `tomato`, `green beans`
- hash bocor antar-split: 0
- label rusak pada verifikasi final: 0

Kitchen test berada di `data/processed/kitchen-top3` dan berisi 120 gambar, termasuk 66 background tanpa salah satu Top 3.

## Smoke training

Smoke satu epoch pada subset 27 gambar berhasil menghasilkan `best.pt` dan `metrics.json`. Ini hanya menguji jalur training, bukan model final.

Validation subset:

- mAP50: 0,0126
- mAP50-95: 0,00523
- recall cucumber: 0
- recall tomato: 0,128
- recall green beans: 0

Kitchen 120 gambar:

- mAP50: 0,1276
- mAP50-95: 0,0760
- cucumber recall: 0,147
- tomato recall: 0,750
- `green beans` tidak memiliki anotasi Kitchen, sehingga tidak ada metrik kelas itu pada test tersebut.

Gate seleksi menolak smoke model karena recall validation ketiga kelas berada di bawah ambang 0,3. Tidak ada model yang dipromosikan ke `models/fruit_detector.pt`.

## Perintah utama

Audit sumber:

```powershell
& '.\.venv\Scripts\python.exe' research\audit_dataset.py --manifest research\download_manifest.yaml --source data\processed\combined-top3-final --output research\reports\audit-combined-top3-final.json
```

Training loop penuh, belum dijalankan sampai selesai di laptop CPU:

```powershell
& '.\.venv\Scripts\python.exe' research\train_loop.py --data data\processed\combined-top3-final\data.yaml --matrix research\experiment_matrix.yaml --output runs\fruitvision-loop --device cpu --max-runs 1
```

Buat file `STOP_TRAINING` di root proyek untuk menghentikan loop di antara eksperimen. Run yang terhenti dicatat sebagai `interrupted` bila menerima `KeyboardInterrupt`.

Evaluasi:

```powershell
& '.\.venv\Scripts\python.exe' research\evaluate.py --model PATH\TO\best.pt --data data\processed\combined-top3-final\data.yaml --output research\reports\evaluation.json --device cpu
```

## Keterbatasan

- Training penuh 80–100 epoch belum selesai. Smoke training bukan bukti akurasi produksi.
- Kitchen tidak memiliki label `green beans`, sehingga kelas itu tidak bisa dinilai lintas-domain dari sumber tersebut.
- Umur dalam hari, penyakit, dan ukuran fisik belum dilatih karena data/provenance yang memenuhi syarat belum tersedia.
- Model tidak dipromosikan otomatis dan tidak dipasang ke aplikasi.
