# Training tomato, pumpkin, lemon

Pipeline ini terpisah dari dataset lama. Dataset baru berada di:

```text
data/processed/tomato-pumpkin-lemon/
```

Urutan kelas wajib:

```text
0 tomato
1 pumpkin
2 lemon
```

## Membuat dataset

Sumber yang diaudit saat ini adalah `data/processed/bangladesh`. Audit tidak
menebak kelas dari nama file atau warna gambar. Kelas lain tidak diubah menjadi
pumpkin atau lemon.

```powershell
& .\.venv\Scripts\python.exe research\tomato_pumpkin_lemon_audit.py `
  --source data\processed\bangladesh `
  --output research\tomato-pumpkin-lemon.audit.json

& .\.venv\Scripts\python.exe research\tomato_pumpkin_lemon_dataset.py `
  --audit research\tomato-pumpkin-lemon.audit.json `
  --output data\processed\tomato-pumpkin-lemon

& .\.venv\Scripts\python.exe research\tomato_pumpkin_lemon_composites.py `
  --dataset data\processed\tomato-pumpkin-lemon `
  --count 1000 `
  --max-synthetic-ratio 0.5
```

Komposit hanya mengambil crop dari `train`. `val`, `test`, dan
`external_holdout` tetap memakai gambar asli. Holdout saat ini masih berasal
dari sumber Bangladesh yang sama, jadi belum merupakan domain eksternal yang
benar-benar independen.

## Training

Validasi tanpa memulai training:

```powershell
& .\.venv\Scripts\python.exe runs\loop\train_tomato_pumpkin_lemon.py `
  --data data\processed\tomato-pumpkin-lemon\data.yaml `
  --epochs 100 --validate-only
```

Untuk training, klik dua kali salah satu preset:

```text
training_presets/train_50_epochs.cmd
training_presets/train_60_epochs.cmd
training_presets/train_70_epochs.cmd
training_presets/train_80_epochs.cmd
training_presets/train_90_epochs.cmd
training_presets/train_100_epochs.cmd
```

`best.pt` dari training belum otomatis menggantikan model aplikasi. Epoch lebih
banyak juga tidak menjamin akurasi.

## Evaluasi dan gate

Evaluasi memakai test real-only, lalu dapat ditambah `external_holdout`:

```powershell
& .\.venv\Scripts\python.exe research\evaluate_tomato_pumpkin_lemon.py `
  --weights path\to\best.pt `
  --data data\processed\tomato-pumpkin-lemon\data.yaml `
  --external-data data\processed\tomato-pumpkin-lemon\data-external.yaml `
  --manifest data\processed\tomato-pumpkin-lemon\split_manifest.json `
  --output research\results\tomato-pumpkin-lemon\evaluation.json
```

Gate tetap `partial` jika kelas wajib tidak memiliki metrik dan `rejected` jika
dukungan atau mAP50 per kelas tidak memenuhi batas. Status `eligible` bukan
bukti bahwa model cocok untuk semua kamera, pencahayaan, atau kondisi buah.

## Batasan aplikasi

Backend menampilkan crop per objek, ID stabil, peringatan objek kecil/tepi/
berdekatan/overlap, dan indikator warna HSV. HSV hanya indikator warna. Ia bukan
diagnosis matang, busuk, atau penyakit tanpa dataset kondisi dan ground truth
yang tervalidasi.
