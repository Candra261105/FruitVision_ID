# Dataset FruitVision ID

Folder ini menjelaskan dataset yang dipakai detector YOLO26n. Gambar dan label dataset tidak disimpan di GitHub karena ukurannya besar.

## Status Rilis & Dataset Siap Pakai

Aplikasi FruitVision ID versi rilis ini telah selesai dan menggunakan model produksi pre-trained **YOLO11m** (35 kelas buah dan sayur, bobot 38,7 MB). Bobot model dapat diunduh otomatis via:

```powershell
py research\download_production_model.py
```

Pengguna tidak perlu mengunduh seluruh file dataset mentah bergiga-giga untuk menjalankan aplikasi web.

Untuk eksperimen training mandiri:
- Paket dataset expansion lengkap siap latih tersedia dalam bentuk arsip: `data/tomato-pumpkin-lemon-expanded.zip`.
- Panduan training di cloud Google Colab dengan GPU T4 gratis dapat dilihat di [`docs/COLAB_EXPANDED_GUIDE.md`](../docs/COLAB_EXPANDED_GUIDE.md).
- Pipeline eksperimen lokal terpisah dengan taxonomy `0 tomato`, `1 pumpkin`, `2 lemon` berada di `data/processed/tomato-pumpkin-lemon`.

## Dataset yang dipakai model epoch 30

Lokasi lokal:

```text
data/processed/combined-top3-cc0-experiment
```

Isi folder:

```text
combined-top3-cc0-experiment/
├── data.yaml
├── data-external.yaml
├── EXPERIMENT_SCOPE.md
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── external_holdout/
└── labels/
    ├── train/
    ├── val/
    ├── test/
    └── external_holdout/
```

Jumlah gambar:

| Split | Jumlah gambar | Fungsi |
|---|---:|---|
| `train` | 3.268 | Melatih model |
| `val` | 319 | Memantau training |
| `test` | 320 | Evaluasi utama yang tidak diubah |
| `external_holdout` | 179 | Evaluasi terpisah dari sumber tambahan |

Kelas detector:

```text
0 cucumber
1 tomato
2 green beans
```

## Dataset targeted

Training lanjutan memakai salinan dataset berikut:

```text
data/processed/combined-top3-cc0-targeted
```

Perubahannya hanya pada `train`:

- 3.268 gambar dasar
- tambahan 690 salinan deterministik gambar yang memiliki label `green beans`
- validation tetap 319 gambar
- test utama tetap 320 gambar
- external holdout tetap 179 gambar

Manifest perubahan ada di:

```text
runs/loop/targeted-dataset-manifest.json
```

Salinan tambahan dibuat untuk mengurangi ketimpangan jumlah contoh `green beans`. Salinan ini bukan gambar baru dari internet.

## Sumber dataset

### 1. Multiple Smartphone Bangladesh Vegetable Detection

Lokasi hasil ekstraksi lokal:

```text
data/processed/bangladesh
```

URL:

https://huggingface.co/datasets/LFTTim/multiple_smartphone_bangladesh_vegetable_detection

Lisensi yang tercatat: **CC BY 4.0**.

Dataset ini menyediakan gambar sayuran dari beberapa jenis smartphone dan menjadi sumber utama untuk kelas `cucumber`, `tomato`, serta `green beans`.

### 2. LVIS Fruits And Vegetables Dataset

Lokasi hasil ekstraksi lokal:

```text
data/processed/lvis-top3
```

Repository:

https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset

Halaman dataset:

https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset

Lisensi yang tercatat: **MIT**.

Hanya gambar yang masuk tiga kelas target yang dipertahankan saat proses penggabungan.

### 3. Tomato Detection Kaggle / greenhouse fruit detection

Lokasi hasil ekstraksi lokal:

```text
data/processed/tomato-cc0-split
```

URL:

https://huggingface.co/datasets/Project-AgML/tomato_detection_kaggle

Sumber asli yang dicatat: Kaggle dan studi greenhouse pada MDPI Agronomy.

Lisensi yang tercatat: **CC0 1.0**.

Pembagian sumber ini:

- 716 gambar dipakai sebagai tambahan train
- 179 gambar disimpan sebagai `external_holdout`

## Cara memakai dataset di komputer lain

Untuk menjalankan aplikasi web, tidak perlu menyalin folder data karena model produksi sudah mandiri.

Jika Anda ingin melanjutkan eksperimen detector berbasis folder lokal:
- Dataset experiment: `FruitVision_ID/data/processed/combined-top3-cc0-experiment`
- Dataset targeted: `FruitVision_ID/data/processed/combined-top3-cc0-targeted`
- Dataset tomato-pumpkin-lemon: `FruitVision_ID/data/processed/tomato-pumpkin-lemon`

`data.yaml` sudah memakai path relatif:

```yaml
path: .
train: images/train
val: images/val
test: images/test
```

Jika dataset berada di folder lain, ubah `path` menjadi lokasi absolut atau jalankan training dengan file YAML yang menunjuk ke lokasi tersebut.

## Cara menyiapkan dataset dari sumber asli

URL dan nama file sumber dicatat di:

```text
research/dataset_registry.yaml
research/download_manifest.yaml
```

Dataset tidak diunduh otomatis oleh project. Setelah mendapatkan file sumber, lakukan audit dan konversi label sebelum training. Jangan memasukkan gambar test atau external holdout ke folder train.

## Catatan GitHub

`.gitignore` mengecualikan:

```text
data/processed/
data/raw/
```

GitHub menyimpan seluruh kode, script training & downloader, gate descriptor, checkpoint, konfigurasi, manifest, dan dokumentasi lengkap. Dataset komposit besar disarankan dilatih di cloud (Colab GPU) menggunakan paket zip yang telah dioptimasi.
