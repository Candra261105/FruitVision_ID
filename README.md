# FruitVision ID

Aplikasi web lokal untuk mendeteksi tiga jenis sayuran dari foto atau kamera:

- `cucumber` atau mentimun
- `tomato` atau tomat
- `green beans` atau buncis

Project ini dibuat untuk tugas PCT106. Backend menggunakan FastAPI dan Ultralytics. Frontend menggunakan React dan Vite. Aplikasi berjalan di komputer lokal, bukan di server publik.

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

Sebagian bobot model yang dipakai project ini sudah disimpan di GitHub karena ukurannya masih di bawah batas file GitHub. Dataset gambar dan checkpoint eksperimen lain tetap disimpan di luar repository karena ukurannya jauh lebih besar.

Struktur minimalnya:

```text
FruitVision_ID/
└── models/
    ├── yolov8s-world.pt
    └── yolo26n-trial.pt
```

Setelah clone, model berikut tersedia langsung:

```text
models/yolo26n-trial.pt
models/yolov8s-world.pt
runs/detect/runs/loop-detector-yolo26n/experiment-512-yolo26n/weights/best.pt
```

### Model YOLO26n untuk uji coba

Bobot lokal yang sudah digunakan pada project ini berada di:

```text
runs\detect\runs\loop-detector-yolo26n\experiment-512-yolo26n\weights\best.pt
```

Salin ke folder model dengan nama berikut:

```powershell
Copy-Item `
  runs\detect\runs\loop-detector-yolo26n\experiment-512-yolo26n\weights\best.pt `
  models\yolo26n-trial.pt
```

Jalankan backend dengan YOLO26n sebagai model uji:

```powershell
$env:FRUITVISION_MODEL_PATH = 'models\yolo26n-trial.pt'
$env:FRUITVISION_ALLOW_UNAPPROVED_CUSTOM = 'true'
$env:FRUITVISION_CONFIDENCE = '0.25'
run_backend.cmd
```

Mode `trial` sengaja menandai bahwa model sedang dicoba. Hasil evaluasi test epoch 30 adalah mAP50 `0,5142`, sehingga model ini belum berstatus tervalidasi untuk menggantikan model fallback.

### Model fallback YOLO-World

Jika `FRUITVISION_MODEL_PATH` tidak diatur, backend mencari:

```text
models\yolov8s-world.pt
```

YOLO-World adalah fallback demo. Ia bukan model custom yang dilatih khusus untuk kondisi kamera pengguna.

## Dataset detector

Dokumentasi dataset ada di:

```text
data\README.md
```

Dataset utama YOLO26n berada di:

```text
data\processed\combined-top3-cc0-experiment
```

Dataset targeted untuk training lanjutan berada di:

```text
data\processed\combined-top3-cc0-targeted
```

Folder dataset dikecualikan dari GitHub karena ukurannya sekitar 2,7 sampai 3,2 GB. Simpan dataset di komputer pengguna, lalu pastikan `data.yaml` menunjuk ke folder yang benar.

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

#### C. Masukkan dataset

Salin dataset targeted ke lokasi berikut:

```text
FruitVision_ID\data\processed\combined-top3-cc0-targeted
```

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
