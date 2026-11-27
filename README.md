# FruitVision ID

Aplikasi web lokal untuk inspeksi dan deteksi hasil panen buah serta sayuran dari foto atau kamera secara real-time.

Versi produksi saat ini menggunakan model **YOLO11m** (`models/fruit_veg_yolo11m.pt`, 38,7 MB, mAP50 92,5%) yang mendukung deteksi **35 komoditas buah dan sayur** dengan pelabelan bahasa Indonesia:

- **Buah**: Apel, Pisang, Ceri, Anggur, Jeruk Hijau, Kiwi, Mangga, Melon, Jeruk, Persik, Pir, Nanas, Delima, Stroberi, Semangka, Lemon.
- **Sayur & Rempah**: Bit Merah, Pare, Labu Air, Kubis, Paprika, Wortel, Kembang Kol, Cabai, Kelapa, Mentimun, Terong, Jahe, Jagung, Okra, Bawang, Kacang Polong, Kentang, Lobak, Tomat, Turnip, Labu.

Project ini dibuat untuk tugas Pengolahan Citra (PCT106). Backend menggunakan FastAPI dan Ultralytics. Frontend menggunakan React dan Vite. Aplikasi berjalan di komputer lokal secara efisien (CPU-friendly).

## Yang perlu disiapkan

- Windows 10 atau 11
- Python 3.11
- `uv`
- Node.js dan npm
- Git, jika project diambil dari GitHub
- Bobot model `.pt`

Pastikan perintah ini bisa dijalankan dari PowerShell:

```powershell
py --version
uv --version
node --version
npm --version
```

## Cara menjalankan setelah clone

Buka PowerShell, lalu masuk ke folder project:

```powershell
cd C:\lokasi\FruitVision_ID
```

Buat environment Python dan pasang dependency backend:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r backend\requirements.txt
```

Pasang dependency frontend:

```powershell
cmd /c npm install --prefix frontend
```

Jalankan backend dan frontend sekaligus:

```powershell
run_all.cmd
```

Alamat yang dipakai:

| Bagian | Alamat |
|---|---|
| Web | http://127.0.0.1:5173 |
| API docs | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |
| Info model | http://127.0.0.1:8000/api/model-info |

Jendela backend dan frontend akan terbuka sebagai dua jendela Command Prompt. Jangan tutup kedua jendela itu selama web digunakan.

## Menyiapkan model

## Menyiapkan model produksi

FruitVision ID menggunakan model deteksi produksi **YOLO11m** (35 kelas buah & sayur) yang telah lolos verifikasi gate (`models/fruit_veg_yolo11m.gate.json`).

### 1. Unduh otomatis bobot model produksi
Jika berkas `models/fruit_veg_yolo11m.pt` belum ada di lokal setelah clone, jalankan skrip pengunduh otomatis:

```powershell
py research\download_production_model.py
```

Skrip ini akan mengunduh model berbobot 38,7 MB langsung dari Hugging Face (`Piyu12/fruit-veg-yolo11m-detector`) ke folder `models/`.

### 2. Struktur folder model

```text
FruitVision_ID/
└── models/
    ├── fruit_veg_yolo11m.pt         # Model produksi 35 kelas (default)
    ├── fruit_veg_yolo11m.gate.json    # Descriptor gate terverifikasi (status: eligible)
    ├── yolov8s-world.pt              # Fallback open-vocabulary demo
    └── yolo26n-trial.pt              # Checkpoint trial legacy (opsional)
```

Backend secara otomatis memilih `models/fruit_veg_yolo11m.pt` sebagai model utama tanpa perlu konfigurasi tambahan.

### 3. Model alternatif / uji coba
Jika ingin menjalankan model kustom atau checkpoint lama secara eksplisit:

```powershell
$env:FRUITVISION_MODEL_PATH = 'models\yolo26n-trial.pt'
$env:FRUITVISION_ALLOW_UNAPPROVED_CUSTOM = 'true'
$env:FRUITVISION_CONFIDENCE = '0.25'
run_backend.cmd
```

## Dataset detector

Dataset eksperimen baru, script audit, komposit train-only, evaluator, dan preset
training dijelaskan di [`docs/tomato-pumpkin-lemon-training.md`](docs/tomato-pumpkin-lemon-training.md).

Dokumentasi dataset ada di:

```text
data\README.md
```

### Model & Dataset Siap Pakai (Versi Selesai)

Aplikasi FruitVision ID versi rilis ini sudah selesai dan siap digunakan langsung:
- Model produksi **35 kelas buah & sayur** (`models/fruit_veg_yolo11m.pt`) langsung aktif dan dapat diunduh otomatis via `py research/download_production_model.py`.
- Tidak perlu mengunduh file dataset besar bergiga-giga untuk menjalankan aplikasi web.

Bagi yang ingin melakukan eksperimen training kustom:
- Paket dataset expansion mandiri tersedia di `data/tomato-pumpkin-lemon-expanded.zip`.
- Panduan training cloud cepat menggunakan GPU T4 gratis di Google Colab tersedia di [`docs/COLAB_EXPANDED_GUIDE.md`](docs/COLAB_EXPANDED_GUIDE.md).
- Dokumentasi struktur dataset lokal dapat dibaca di [`data/README.md`](data/README.md).

## Melanjutkan training di komputer lain

Training bisa dilanjutkan di komputer teman tanpa memulai dari `yolo26n.pt`. Yang dipindahkan adalah source code project, dataset YOLO, dan checkpoint model `best.pt` dari training sebelumnya.

### Perbedaan `best.pt` dan `last.pt`

| File | Kegunaan |
|---|---|
| `best.pt` | Memulai fine-tuning lanjutan dari bobot terbaik yang sudah didapat |
| `last.pt` | Melanjutkan sesi training yang terhenti, termasuk state optimizer dan nomor epoch |

Checkpoint dasar YOLO26n epoch 30 yang dipakai project ini berada di:

```text
runs/detect/runs/loop-detector-yolo26n/experiment-512-yolo26n/weights/best.pt
```

Checkpoint tersebut sudah ikut di repository. Dataset tetap perlu disalin dari penyimpanan terpisah karena folder `data/processed/` tidak masuk GitHub.

### Cara 1: memakai VS Code

#### A. Clone project

Buka VS Code, tekan `Ctrl+Shift+P`, pilih **Git: Clone**, masukkan URL repository GitHub, lalu pilih **Open** pada folder project.

Atau jalankan dari terminal:

```powershell
git clone https://github.com/Candra261105/FruitVision_ID.git
cd FruitVision_ID
code .
```

Repository ini bersifat public. Siapa pun yang memiliki link dapat melakukan clone.

#### B. Siapkan Python di terminal VS Code

Buka terminal dengan `Ctrl+`` lalu jalankan:

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r backend\requirements.txt
```

Di VS Code, tekan `Ctrl+Shift+P`, pilih **Python: Select Interpreter**, lalu pilih:

```text
.venv\Scripts\python.exe
```

#### C. Menyiapkan model produksi & dataset

Untuk menjalankan aplikasi secara langsung, unduh bobot model produksi:

```powershell
py research\download_production_model.py
```

Jika ingin melatih ulang detector, paket dataset mandiri sudah disediakan di `data/tomato-pumpkin-lemon-expanded.zip`. Ikuti panduan Google Colab T4 GPU di [`docs/COLAB_EXPANDED_GUIDE.md`](docs/COLAB_EXPANDED_GUIDE.md) agar proses training selesai cepat dan tidak membebani CPU komputer.

Jika menggunakan dataset lokal lama untuk eksperimen:
- Folder dataset: `data\processed\combined-top3-cc0-targeted`
- Manifest perubahan: `runs\loop\targeted-dataset-manifest.json`

Folder tersebut harus memiliki `data.yaml`, folder `images`, dan folder `labels` dengan split `train`, `val`, `test`, serta `external_holdout`.

Pastikan `data.yaml` berisi tiga kelas berikut:

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  - cucumber
  - tomato
  - green beans
```

#### D. Periksa checkpoint

Checkpoint dasar sudah tersedia di lokasi yang dicari script:

```text
runs\detect\runs\loop-detector-yolo26n\experiment-512-yolo26n\weights\best.pt
```

Jika ingin menyimpan checkpoint di lokasi lain, gunakan environment variable. Source code tidak perlu diubah:

```powershell
$env:FRUITVISION_BASE_WEIGHTS = 'D:\FruitVisionTransfer\best.pt'
```

Jika file itu dipindahkan ke lokasi lain, ubah environment variable tersebut sesuai lokasi baru.

#### E. Jalankan training lanjutan

Jalankan dari folder root project di terminal VS Code:

```powershell
.\.venv\Scripts\python.exe runs\loop\train_yolo26n_targeted.py
```

Script akan memeriksa dataset dan nama kelas, membaca `best.pt`, lalu melakukan fine-tuning selama 20 epoch tambahan. Ukuran gambar yang dipakai adalah `512`. GPU CUDA akan dipakai jika tersedia, sedangkan komputer tanpa CUDA akan memakai CPU.

Untuk menambah jumlah epoch, buka file:

```text
runs\loop\train_yolo26n_targeted.py
```

Ubah nilai berikut, misalnya dari 20 menjadi 100:

```python
epochs=100
```

Kemudian jalankan script yang sama. Hasil training baru akan dibuat di folder `runs\detect` dan tidak menimpa checkpoint lama selama nama run tidak dibuat sama.

Hasil training berada di:

```text
runs\detect\runs\loop-detector-yolo26n-targeted\continuation-512-yolo26n-from-epoch30\
```

Checkpoint hasilnya:

```text
weights\best.pt
weights\last.pt
```

Jangan menutup terminal selama proses berjalan. Jika training dihentikan, simpan folder hasil tersebut, terutama file `weights\last.pt`.

### Preset training 50 sampai 100 epoch

Repository ini menyediakan preset berikut di folder `training_presets`:

| File | Jumlah epoch dalam satu sesi fine-tuning |
|---|---:|
| `train_50_epochs.cmd` | 50 |
| `train_60_epochs.cmd` | 60 |
| `train_70_epochs.cmd` | 70 |
| `train_80_epochs.cmd` | 80 |
| `train_90_epochs.cmd` | 90 |
| `train_100_epochs.cmd` | 100 |

Pastikan dataset dan `.venv` sudah siap. Dari VS Code, jalankan salah satu perintah berikut di terminal project:

```powershell
cmd /c training_presets\train_50_epochs.cmd
cmd /c training_presets\train_60_epochs.cmd
cmd /c training_presets\train_70_epochs.cmd
cmd /c training_presets\train_80_epochs.cmd
cmd /c training_presets\train_90_epochs.cmd
cmd /c training_presets\train_100_epochs.cmd
```

Atau klik dua kali file `.cmd` yang diinginkan dari File Explorer. Setiap preset memulai fine-tuning dari checkpoint epoch 30 yang ada di repository, lalu membuat folder hasil sendiri. Preset tidak menjalankan keenam training sekaligus.

Untuk jumlah lain, misalnya 110 epoch, gunakan runner yang sama:

```powershell
& .\.venv\Scripts\python.exe runs\loop\train_targeted_epochs.py --epochs 110
```

Nilai `--epochs` adalah jumlah epoch pada sesi fine-tuning dari checkpoint awal, bukan jaminan bahwa model langsung lebih akurat. Evaluasi `best.pt` pada test dan `external_holdout` sebelum mengganti model web.

Jika training terhenti, jangan menjalankan ulang preset yang sama dengan `--resume` secara otomatis. Simpan dahulu folder hasil dan gunakan `weights\last.pt` hanya untuk melanjutkan run yang sama setelah memastikan checkpoint tersebut utuh.

### Cara 2: memakai PowerShell tanpa VS Code

VS Code hanya membantu membuka project dan memantau proses. Training tetap dijalankan oleh Python, jadi bisa dilakukan langsung dari PowerShell:

```powershell
cd C:\lokasi\FruitVision_ID
$env:FRUITVISION_BASE_WEIGHTS = 'C:\Transfer\best.pt'
.\.venv\Scripts\python.exe runs\loop\train_yolo26n_targeted.py
```

Komputer tanpa GPU NVIDIA tetap bisa menjalankan training di CPU, tetapi waktunya lebih lama. Jika memakai GPU NVIDIA, pastikan PyTorch di environment tersebut mendukung CUDA.

### Cara 3: resume sesi yang benar-benar terhenti

Gunakan `last.pt` jika ingin melanjutkan sesi training yang terputus. Ultralytics akan memuat bobot, optimizer, scheduler, dan nomor epoch dari checkpoint tersebut.

Contoh Python:

```python
from ultralytics import YOLO

model = YOLO(r'C:\Transfer\last.pt')
model.train(resume=True)
```

Atau melalui CLI:

```powershell
yolo train resume model="C:\Transfer\last.pt"
```

`resume=True` dipakai untuk melanjutkan run lama. Jangan memakai `resume=True` pada `best.pt` jika tujuan Anda adalah fine-tuning dengan dataset targeted dan konfigurasi baru. Untuk project ini, gunakan `train_yolo26n_targeted.py` dengan `best.pt`.

### Mengecek checkpoint sebelum training

```powershell
Test-Path $env:FRUITVISION_BASE_WEIGHTS
Get-Item $env:FRUITVISION_BASE_WEIGHTS | Select-Object FullName,Length
```

Saat script dimulai, terminal menampilkan baris `BASE_WEIGHTS` dan checksum SHA-256. Simpan baris itu dalam laporan agar sumber checkpoint dapat dilacak.

Setelah training selesai, jangan langsung mengganti model web. Evaluasi dulu `weights\best.pt` pada test utama dan external holdout. Model hasil lanjutan masih berstatus kandidat sampai metriknya diperiksa.

## Penggunaan API

### Cek status

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/api/model-info -UseBasicParsing
```

### Deteksi gambar

Endpoint deteksi menerima satu file gambar:

```powershell
curl.exe -X POST `
  -F "file=@C:\foto\sayuran.jpg" `
  http://127.0.0.1:8000/api/predict/image
```

Web menyediakan jalur upload gambar dan kamera. Detector menghasilkan bounding box untuk objek yang terdeteksi.

## Pengujian project

Backend:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests -q
cd ..
```

Frontend:

```powershell
cmd /c npm test --prefix frontend -- --run
cmd /c npm run build --prefix frontend
```

## Struktur folder penting

```text
backend/                         API FastAPI dan service deteksi
frontend/                        Tampilan React
models/                          Bobot model lokal
data/README.md                   Dokumentasi dataset
data/processed/                  Dataset lokal, tidak masuk Git
research/                        Script audit dataset dan training
runs/                            Script dan metadata run yang dipilih
training_presets/                Preset training 50, 60, 70, 80, 90, dan 100 epoch
run_all.cmd                      Menjalankan backend dan frontend
run_backend.cmd                  Menjalankan backend saja
run_frontend.cmd                 Menjalankan frontend saja
```

## Batasan hasil deteksi

- Model hanya mengenali tiga kelas detector yang disebutkan di atas.
- Hasil model dapat berubah pada foto dengan pencahayaan, jarak, latar, atau susunan objek yang berbeda dari dataset.
- Model tidak mendiagnosis penyakit, kualitas, tingkat kematangan, atau keamanan pangan secara otomatis.
- Bounding box bukan segmentation mask.
- Model trial YOLO26n belum boleh disebut tervalidasi produksi sebelum melewati evaluasi gate.

## Sumber dan lisensi

Sumber dataset, URL, lisensi, dan pembagian data dicatat di:

```text
research\dataset_registry.yaml
research\download_manifest.yaml
data\README.md
```

Kode project dan setiap dataset memiliki ketentuan lisensi yang berbeda. Baca lisensi sumber sebelum membagikan ulang gambar atau bobot model.

## Batasan penyimpanan GitHub

Repository ini sudah berisi kode, dokumentasi, konfigurasi, manifest, laporan audit, dan beberapa bobot model. Dataset gambar, file Parquet, cache, serta hasil eksperimen besar sengaja tidak di-push. Ukuran data lokal yang dikecualikan sekitar 18,6 GB dan tidak sesuai dengan batas GitHub biasa.

Untuk training di komputer lain, siapkan dataset sesuai `data/README.md`, lalu jalankan script training dari folder project. Untuk menjalankan web, model trial sudah tersedia di `models/yolo26n-trial.pt`.
