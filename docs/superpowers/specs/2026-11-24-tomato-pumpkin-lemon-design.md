# Desain Dataset Tomato, Pumpkin, dan Lemon

## Maksud

Membuat dataset detector baru untuk tiga kelas yang benar-benar sesuai dengan
aplikasi:

```text
tomato
pumpkin
lemon
```

Dataset lama tidak ditimpa. Dataset baru dipakai untuk eksperimen dan evaluasi
terpisah agar hasil model lama tetap bisa dibandingkan.

## Tujuan

- Memakai gambar yang sudah ada di project sebagai sumber utama.
- Memperbaiki atau membuat bounding box hanya setelah objek diverifikasi.
- Menambah variasi objek berdekatan, menempel, bertumpuk, kecil, gelap, terang,
  blur, dan noise melalui komposit sintetis dari objek nyata.
- Menyediakan file `data.yaml`, laporan dataset, manifest sintetis, dan runner
  training 50, 60, 70, 80, 90, 100 epoch serta angka lain.
- Menguji model pada gambar asli yang tidak dipakai untuk membuat komposit.

## Batasan dan kejujuran data

- Label `cucumber` dan `green beans` tidak boleh diganti menjadi `pumpkin` atau
  `lemon`. Pergantian seperti itu akan menghasilkan label palsu.
- Folder project akan dipindai untuk mencari gambar dan label yang benar-benar
  relevan. Jika tidak ada sumber asli untuk suatu kelas, kelas tersebut tidak
  boleh diisi dengan tebakan atau gambar buatan tanpa objek sumber.
- Komposit sintetis bukan bukti bahwa model akurat di dunia nyata. Evaluasi
  utama tetap memakai gambar asli pada `val`, `test`, dan bila tersedia
  `external_holdout`.
- Akurasi tinggi tidak dijanjikan sebelum metrik per kelas, confusion matrix,
  dan uji pada foto baru selesai.

## Pendekatan yang dipilih

### 1. Gambar existing + perbaikan label

Pipeline akan:

1. Memindai seluruh folder dataset yang sudah ada.
2. Membaca format YOLO dan memeriksa pasangan gambar-label.
3. Mencari sumber objek tomato, pumpkin, dan lemon yang bisa diverifikasi.
4. Memeriksa koordinat bounding box, ukuran objek, label kosong, file rusak,
   dan duplikasi gambar.
5. Mempertahankan gambar asli yang valid dan menulis ulang label ke skema tiga
   kelas baru.
6. Membagi data asli menjadi `train`, `val`, dan `test` tanpa mencampur turunan
   gambar yang sama ke split berbeda.

Jika label lama hanya menyediakan tomato, maka tomato tetap bisa dipakai,
sedangkan pumpkin dan lemon menunggu sumber asli yang ditemukan di project.

### 3. Komposit sintetis dari objek nyata

Objek yang sudah diverifikasi akan dipotong dari gambar sumber dan ditempatkan
ke latar lain atau kanvas komposit. Bounding box komposit ditulis otomatis dari
posisi objek yang ditempel.

Variasi yang dibuat hanya untuk `train`:

- posisi objek berjauhan dan berdekatan,
- objek menempel atau sebagian saling menutupi,
- skala kecil dan besar,
- rotasi ringan dan perubahan perspektif terbatas,
- pencahayaan gelap/terang,
- blur gerak dan blur fokus,
- noise kamera,
- perubahan warna yang tidak mengubah kelas buah.

Komposit diberi manifest yang mencatat sumber gambar, kelas, transformasi,
posisi, dan seed agar dapat diulang. Jumlah komposit dibatasi agar dataset
sintetis tidak mengalahkan gambar asli.

## Struktur output

```text
data/processed/tomato-pumpkin-lemon/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── synthetic_manifest.json
├── dataset_report.json
└── README.md
```

Data sintetis hanya boleh berada di `train`. `val` dan `test` memakai data asli
yang dipisahkan sebelum augmentasi dan komposit dibuat.

## Training

File training akan menerima parameter epoch, misalnya:

```powershell
& .\.venv\Scripts\python.exe runs\loop\train_tomato_pumpkin_lemon.py --epochs 100
```

Preset `.cmd` akan disediakan untuk 50, 60, 70, 80, 90, dan 100 epoch. Model
awal, ukuran gambar, batch, device, dan augmentasi akan mengikuti kemampuan
komputer project. Training tidak otomatis mengganti model web.

## Integrasi aplikasi

Setelah dataset tersedia dan model melewati evaluasi minimum:

- daftar kelas detector diubah menjadi tomato, pumpkin, lemon,
- hasil setiap bounding box dirender sebagai crop objek,
- pengguna dapat memilih objek tertentu,
- pemeriksaan kondisi ditampilkan per objek,
- estimasi warna berbasis HSV ditampilkan sebagai persentase mentah/matang/
  busuk bila aturan dan data pendukung tersedia,
- objek yang menyentuh tepi, kecil, terlalu dekat, atau saling tumpang tindih
  diberi peringatan verifikasi.

Integrasi UI tidak boleh menyamarkan estimasi HSV sebagai label kematangan yang
sudah tervalidasi jika belum ada model atau aturan yang diuji untuk ketiga
kelas.

## Pemeriksaan penerimaan

- Dataset lama tetap utuh.
- Setiap gambar yang dipakai memiliki label valid dan kelas sesuai isi gambar.
- Tidak ada gambar turunan yang bocor dari `train` ke `val` atau `test`.
- Manifest sintetis dapat dibaca dan jumlah objeknya cocok dengan label.
- Script training dapat dijalankan dengan epoch preset dan angka bebas.
- Metrik dilaporkan per kelas, bukan hanya satu angka rata-rata.
- Backend dan frontend tetap lulus test setelah integrasi.

## Hal yang masih perlu dipastikan dari isi folder

Builder akan memeriksa apakah sumber asli pumpkin dan lemon memang tersedia.
Jika salah satu tidak ada, hasil tahap dataset akan menyatakan kelas itu belum
memiliki sumber valid dan tidak akan mengarang data untuk menutup kekosongan.
