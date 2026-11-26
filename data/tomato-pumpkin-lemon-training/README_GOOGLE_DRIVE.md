# Paket Training Tomato, Pumpkin, Lemon

Paket ini mandiri untuk training detector tiga kelas:

```text
0 tomato
1 pumpkin
2 lemon
```

Dataset lama di project tidak ikut ditimpa. Paket ini berisi salinan dataset
baru, script training, evaluator, dan launcher Windows.

## Pilihan penggunaan

Untuk teman yang biasa bekerja di VS Code, gunakan **mode hybrid notebook** di
bawah. Launcher `.cmd` tetap tersedia sebagai cara cadangan yang lebih singkat.

## Cara memakai dari Google Drive

1. Unduh seluruh folder `tomato-pumpkin-lemon-training` ke komputer. Jangan
   hanya mengunduh folder `launchers`.
2. Pastikan Python 3.10 atau lebih baru tersedia dan perintah `py` dapat
   dijalankan dari Command Prompt.
3. Buka Command Prompt di folder paket, lalu pasang dependency:

   ```bat
   py -m pip install -r requirements-training.txt
   ```

   Jika memakai virtual environment, perintah yang disarankan:

   ```bat
   py -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements-training.txt
   ```

4. Buka folder `launchers`.
5. Klik dua kali salah satu file:

   - `train_50_epochs.cmd`
   - `train_60_epochs.cmd`
   - `train_70_epochs.cmd`
   - `train_80_epochs.cmd`
   - `train_90_epochs.cmd`
   - `train_100_epochs.cmd`

Launcher memilih `.venv` bila folder tersebut ada. Jika tidak ada, launcher
menggunakan Python dari perintah `py`. Semua launcher menunjuk ke
`dataset\data.yaml` di paket ini, bukan ke folder project lain.

## Mode hybrid di VS Code

Paket juga menyediakan notebook agar alurnya mirip proyek
`pepper_detection_disease.ipynb`, tetapi path dataset dibuat relatif dan
training tetap memanggil script resmi.

Buka folder paket ini di VS Code, lalu pasang dependency pada interpreter yang
akan dipakai notebook:

```bat
py -m pip install -r requirements-training.txt
```

Jika teman memakai `.venv` di luar folder paket, itu aman. Contoh:

```text
C:\Users\dragon\venvs\fruitvision\.venv
C:\Users\dragon\Downloads\...\tomato-pumpkin-lemon-training
```

`.venv` di luar tidak perlu dipindahkan ke dalam paket. Di VS Code pilih:

```text
Python: Select Interpreter
```

Lalu pilih Python dari `.venv` luar tersebut dan pilih kernel yang sama untuk
notebook. Notebook memakai `sys.executable` dari kernel aktif, sehingga
dependency dan GPU yang dipakai sama dengan interpreter pilihan teman.

Jangan menjalankan notebook dengan kernel Python yang berbeda dari interpreter
tempat dependency dipasang.

Jika menjalankan launcher dari Command Prompt, arahkan launcher ke Python dari
`.venv` luar tanpa memindahkannya:

```bat
set FRUITVISION_PYTHON=C:\Users\dragon\venvs\fruitvision\Scripts\python.exe
run_train_auto.cmd
```

Jika environment dibuat dengan `uv`, instal dependency langsung ke environment
tersebut:

```bat
uv pip install --python C:\Users\dragon\venvs\fruitvision\Scripts\python.exe -r requirements-training.txt
```

Urutan notebook:

```text
notebooks\01_dataset_inspection.ipynb
notebooks\02_train_tomato_pumpkin_lemon.ipynb
notebooks\03_evaluate_model.ipynb
```

Notebook 01 hanya membaca dan menampilkan dataset. Notebook 02 menjalankan
runner adaptive. Notebook 03 baru dijalankan setelah `best.pt` tersedia.

## Training adaptive di laptop berbeda

Cara paling sederhana adalah menjalankan `run_train_auto.cmd` dari folder paket:

```bat
run_train_auto.cmd
```

Runner mendeteksi `torch.cuda.is_available()`:

| Perangkat | Default | Profil |
|---|---:|---|
| NVIDIA CUDA, termasuk RTX 2050 | 100 epoch | `imgsz=640`, `batch=2`, `workers=2`, `device=0` |
| AMD/CPU tanpa CUDA | 10 epoch | `imgsz=416`, `batch=1`, `workers=0`, `device=cpu` |

Profil CPU sengaja ringan supaya laptop lama tidak langsung dibebani training
100 epoch. Untuk mengatur manual:

```bat
run_train_auto.cmd --epochs 1
run_train_auto.cmd --epochs 100 --batch 1
```

Checkpoint disimpan setiap epoch di:

```text
runs\adaptive\<nama-run>\weights\last.pt
runs\adaptive\<nama-run>\weights\best.pt
```

Jika proses terhenti, lanjutkan dari checkpoint terakhir:

```bat
run_train_auto.cmd --resume "runs\adaptive\adaptive-10ep\weights\last.pt"
```

`test` dan `external_holdout` tidak ikut training. Training juga tidak
mengganti model aplikasi secara otomatis.

## Hasil training

Hasil dari launcher lama disimpan di:

```text
runs\tomato-pumpkin-lemon-<epoch>ep\weights\best.pt
```

Nama folder dapat memiliki tambahan bila nama tersebut sudah pernah dipakai.
File `best.pt` belum disertakan dalam paket ini karena training belum
dijalankan. Jumlah epoch juga tidak menjamin akurasi tertentu.

Untuk runner adaptive, hasil disimpan di `runs\adaptive\<nama-run>\weights\`.

Setelah training selesai, evaluasi checkpoint pada test dan external holdout
dapat dilakukan dengan `training\evaluate_tomato_pumpkin_lemon.py`. Evaluasi
harus melihat metrik per kelas, bukan hanya satu angka rata-rata. Model baru
tidak otomatis menggantikan model aplikasi.

Contoh evaluasi lengkap dari root paket:

```bat
py training\evaluate_tomato_pumpkin_lemon.py ^
  --weights runs\tomato-pumpkin-lemon-100ep\weights\best.pt ^
  --data dataset\data.yaml ^
  --external-data dataset\data-external.yaml ^
  --manifest dataset\split_manifest.json ^
  --output runs\tomato-pumpkin-lemon-100ep\evaluation.json
```

Ganti nama folder hasil sesuai epoch yang benar-benar dijalankan.

## Isi penting paket

```text
dataset\data.yaml                 konfigurasi kelas dan split
dataset\images\                   gambar train/val/test/external_holdout
dataset\labels\                   label YOLO untuk setiap gambar
training\train_tomato_pumpkin_lemon.py
training\evaluate_tomato_pumpkin_lemon.py
launchers\train_*_epochs.cmd
requirements-training.txt
```

Komposit sintetis, bila ada, hanya berada pada `dataset\images\train` dan
`dataset\labels\train`. `val`, `test`, dan `external_holdout` tetap dipakai
untuk evaluasi gambar asli sesuai laporan dataset.

## Jika launcher berhenti

- Pesan dependency belum tersedia: jalankan perintah instalasi pada langkah 3.
- Pesan file tidak ditemukan: pastikan folder paket diunduh lengkap dan
  `dataset\data.yaml` ada.
- Training dapat memakan waktu lama, terutama pada CPU. Jangan menutup jendela
  selama proses masih berjalan.
