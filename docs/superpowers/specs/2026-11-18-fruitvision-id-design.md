# Desain FruitVision ID

Tanggal: 18 November 2026  
Status: rancangan untuk persetujuan, keputusan pengguna sudah dikumpulkan,
belum diimplementasikan

## 1. Maksud dan pengguna

FruitVision ID adalah web lokal untuk menguji pengolahan citra buah yang umum
di Indonesia melalui kamera laptop atau unggah gambar. Sistem harus mampu
menjawab, per buah:

- buah apa yang terlihat;
- berapa objek buah yang terdeteksi, termasuk saat berdempetan;
- tingkat kematangan dan kualitas visual;
- perkiraan rentang hari menuju matang jika data kalibrasinya tersedia;
- masalah visual yang mungkin terlihat dan alasan singkatnya.

Target pertama adalah MVP terlatih untuk tugas kuliah PCT106 yang bisa
didemonstrasikan secara offline, bukan alat inspeksi pangan komersial atau
diagnosis penyakit tanaman yang definitif.

## 2. Batasan yang memengaruhi desain

1. Dataset publik terbesar umumnya hanya berisi satu buah terisolasi dan label
   jenis buah. Dataset itu membantu klasifikasi, tetapi tidak cukup untuk
   menghitung buah yang berdempetan.
2. Label ripe/unripe/rotten tidak sama dengan umur. Tidak ada dasar jujur untuk
   mengeluarkan angka hari dari sebuah foto jika tidak ada data longitudinal
   yang menghubungkan foto dengan tanggal matang.
3. Dataset dengan nama Indonesia belum tentu berasal dari Indonesia, memiliki
   label yang baik, atau memiliki lisensi yang boleh dipakai ulang.
4. Laptop target Ryzen 5 3500U dengan Vega 8 dan RAM 16 GB lebih cocok untuk
   inferensi foto atau snapshot kamera daripada streaming video beresolusi
   tinggi.
5. Beberapa repositori populer tidak mencantumkan lisensi. Kode atau data dari
   sumber seperti itu hanya dijadikan referensi sampai izin penggunaannya jelas.
6. Pengguna tidak ingin mengambil foto sendiri. Set uji akan memakai gambar
   publik yang disisihkan berdasarkan sumber atau kelompok gambar. Ini dapat
   menguji generalisasi antar-sumber, tetapi tidak membuktikan performa pada
   kondisi kamera laptop pengguna.

## 3. Tujuan, bukan tujuan, dan anti-tujuan

### Tujuan

- Memulai dari lima buah prioritas: pisang, mangga, pepaya, alpukat, dan jeruk.
- Menyediakan jalur ekspansi untuk nanas, jambu, salak, rambutan, durian,
  semangka, melon, tomat, cabai, dan kelapa sawit.
- Menggunakan instance segmentation atau deteksi yang mempertahankan objek
  terpisah, bukan sekadar satu label untuk seluruh gambar.
- Menggabungkan data publik yang lisensinya jelas. Data lokal tidak menjadi
  syarat MVP karena pengguna memilih memakai gambar internet yang sudah ada.
- Jika label kematangan untuk salah satu dari lima jenis tidak cukup, sistem
  tetap mendeteksi jenisnya dan memberi status bahwa kematangan belum tersedia.
- Menampilkan confidence, sumber keputusan, dan peringatan jika input di luar
  kelas yang dilatih.
- Menguji beberapa konfigurasi model secara berulang pada split yang sama dan
  memilih berdasarkan metrik, bukan berdasarkan satu contoh yang terlihat bagus.
- Menyediakan laporan reproduksi: sumber data, checksum, lisensi, konfigurasi,
  hasil evaluasi, dan contoh kegagalan.

### Bukan tujuan

- Menjamin prediksi tanggal matang yang pasti dari satu foto.
- Mengklaim penyakit tertentu hanya berdasarkan warna atau bercak yang terlihat.
- Menggabungkan semua dataset publik tanpa audit label, deduplikasi, dan audit
  lisensi.
- Membuat layanan cloud atau aplikasi mobile pada tahap pertama.

### Anti-tujuan

- Tidak menampilkan angka hari jika model belum dikalibrasi dengan data waktu.
- Tidak menampilkan confidence palsu yang berasal dari heuristik sederhana.
- Tidak menyebut "penyakit X" bila label dataset hanya menyatakan rusak,
  bercak, busuk, atau tidak sehat.

## 4. Pendekatan yang dipertimbangkan

### Pendekatan 1: klasifikasi satu gambar

Model klasifikasi memprediksi kelas dari seluruh gambar atau crop manual.
Pendekatan ini murah dan cocok untuk baseline, tetapi gagal menghitung banyak
buah, tidak menangani buah berdempetan, dan tidak menunjukkan lokasi objek.
Dipakai sebagai baseline saja.

### Pendekatan 2: satu model deteksi dengan kelas gabungan

Kelas model dibuat seperti `mango_ripe`, `mango_unripe`, dan
`mango_rotten`. Ini mudah dipasang, tetapi jumlah kelas cepat membesar,
ketimpangan data meningkat, dan sulit menambah label baru tanpa melatih ulang
seluruh kombinasi.

### Pendekatan 3: pipeline hybrid bertahap, pilihan desain

1. Model instance segmentation mendeteksi jenis buah dan memisahkan setiap
   objek.
2. Crop/mask tiap objek masuk ke classifier kualitas dan kematangan.
3. Modul kalibrasi mengubah tingkat kematangan menjadi interval hari hanya
   untuk buah yang memiliki data longitudinal.
4. Modul aturan menghasilkan penjelasan dari label yang terlihat, bukan dari
   teks bebas model bahasa.

Pendekatan ini dipilih karena bisa memakai dataset yang berbeda fungsi,
menangani buah berdempetan, mengisolasi kesalahan tiap tahap, dan dapat
menambah jenis buah tanpa merombak antarmuka.

## 5. Riset dataset awal

Status berikut adalah hasil pencarian awal pada 18 November 2026. Semua sumber
harus diaudit ulang oleh skrip sebelum dipakai untuk pelatihan.

| Sumber | Isi yang terlihat | Peran | Lisensi/status | Keputusan awal |
|---|---|---|---|---|
| [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) | Sekitar 200.201 gambar, 278 kelas buah/sayur/kacang/jamur | Klasifikasi jenis dan pretraining | CC BY-SA 4.0 | Dipakai sebagai data pendukung, bukan deteksi kerumunan |
| [Fruit Ripeness Object Detection](https://www.kaggle.com/datasets/killa92/fruit-ripeness-object-detection-dataset) | Sekitar 1,17 GB, YOLO bounding box, pisang Lakatan, mangga Carabao, apel, melon, pir, jeruk, persik, anggur, empat tahap | Baseline deteksi + kematangan | Apache 2.0 | Kandidat utama setelah audit isi dan label |
| [Fruit Ripeness: Unripe, Ripe, Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten) | Sekitar 3,95 GB, pisang, apel, jeruk, tiga status | Classifier kualitas | CC BY-SA 4.0 | Dipakai untuk baseline kualitas, cek duplikasi |
| [Mango/Banana Ripe-Unripe di Hugging Face](https://huggingface.co/datasets/darthraider/fruit-ripeness-detection-dataset) | Sekitar 5.000 gambar, kondisi luar ruang, mangga dan pisang | Classifier atau crop detector | Apache 2.0 di kartu dataset, berasal dari Mendeley | Kandidat, provenance dan versi harus dicatat |
| [Mango/Banana Mendeley](https://data.mendeley.com/datasets/y3649cmgg/3) | Ripe/unripe dan anotasi YOLO menurut katalog sumber | Deteksi dua buah | Tautan saat audit mengembalikan 404 | Jangan unduh sebelum sumber resmi ditemukan ulang |
| [Oil palm ripeness AgML](https://huggingface.co/datasets/Project-AgML/oil_palm_fruit_ripeness_classification) | 466 gambar, Damaged, Empty, Overripe, Ripe, Unripe | Buah sawit, relevan untuk Indonesia | CC BY 4.0 | Kandidat kualitas kecil, perlu penyeimbangan |
| [Fruit Ripeness 12 classes](https://huggingface.co/datasets/Carlosmg23/fruit-ripeness-12clases) | 8.256 crop, apel/pisang/mangga, unripe/ripe/overripe/rotten, imbalance maksimum 17,2x | Classifier tahap dan cacat | Kartu dataset perlu dibaca lengkap | Dipakai hanya setelah audit turunan dan imbalance |
| [Five tropical fruit ripeness](https://github.com/xuetong932-hub/Five-Fruit-Ripeness-Image-Dataset-with-Monocular-Depth-Maps) | Klaim 21.952 gambar, pisang, durian, jambu, mangga, pepaya, ripe/unripe + depth | Kematangan dan eksperimen depth | Lisensi belum terlihat di hasil awal | Kandidat riset, belum untuk distribusi |
| [Fruit Ripeness Detection](https://github.com/anujgoenka9/Fruit-Ripeness-Detection) | Apel, pisang, jeruk, delima, mangga, pepaya, raw/ripe/rotten | Baseline klasifikasi | MIT | Rujukan kode dan baseline, bukan bukti akurasi baru |
| [Carabao/Lakatan/Red Lady app](https://github.com/Matt-CRL/fruit-ripeness-detector) | Aplikasi offline untuk mangga, pisang, pepaya | Referensi produk dan kelas lokal Asia | Lisensi/data belum diaudit | Ambil ide alur, jangan menyalin data sebelum jelas |
| [Fruit detection YOLO POS](https://github.com/Kavan-Patel/Fruits-And-Vegetable-Detection-for-POS-with-Deep-Learning) | Deteksi banyak kelas untuk skenario kasir | Referensi deteksi dan deployment | Lisensi tidak terlihat | Referensi saja |
| [Mango disease YOLOv8](https://github.com/Muthu39/mango-fruit-disease-detection-using-yolov8-) | Model dan web Flask untuk penyakit mangga | Referensi diagnosis visual | Lisensi tidak terlihat | Referensi saja, label penyakit harus diverifikasi |
| [Pomegranate stage YOLO](https://github.com/tejteja54321/Pomogranate-Fruit-Stage-Detection-YOLO) | Tahap immature/mature/ripe pada delima | Referensi label tahap | Lisensi tidak terlihat | Referensi saja |
| [Dragon fruit ripeness](https://github.com/share2code99/dragon_fruit_ripeness_detection_yolov5_spdconv) | Deteksi kematangan buah naga | Referensi buah tropis | Lisensi tidak terlihat | Referensi saja |

Catatan: nama seperti Carabao, Lakatan, Red Lady, dan dataset sawit lebih
dekat dengan konteks Asia Tenggara atau Indonesia, tetapi itu tetap tidak
membuktikan seluruh gambar berasal dari Indonesia. Metadata asal gambar harus
disimpan sebagai field terpisah.

## 6. Estimasi hari menuju matang

Estimasi hari bukan bagian aktif dari MVP. Label `ripe` atau `unripe` tidak
memberikan tanggal, sehingga tidak boleh diubah menjadi angka hari dengan
aturan buatan.

Karena pengguna tidak ingin mengambil foto sendiri, modul hari hanya boleh
diaktifkan jika sumber internet menyediakan pasangan foto dan waktu yang dapat
diverifikasi, misalnya tanggal pengambilan dan tanggal buah mencapai tahap
siap makan. Dataset ripe/unripe biasa tidak memenuhi syarat ini.

Pada MVP, web menampilkan tahap kematangan dan teks "estimasi hari belum
terkalibrasi". Jika fase kedua memiliki data waktu yang cukup, interval dapat
dibuat dengan quantile regression atau conformal interval per jenis buah, lalu
ditulis sebagai "perkiraan 2--4 hari, keyakinan rendah".

## 7. Arsitektur teknis yang diusulkan

### Backend Python

- FastAPI dengan Pydantic untuk request/response.
- OpenCV dan Pillow untuk decode, resize, warna, dan overlay.
- Ultralytics YOLO sebagai baseline instance segmentation ringan. Versi model
  final dipilih lewat benchmark, bukan diasumsikan sejak awal.
- PyTorch untuk training/inference.
- scikit-learn atau modul statistik kecil untuk kalibrasi hari dan metrik.
- Model di-load sekali saat startup, bukan setiap request.
- Endpoint awal:
  - `GET /health`
  - `GET /api/model-info`
  - `POST /api/predict/image`
  - `POST /api/predict/frame`
  - `GET /api/calibration-status`

Response per objek minimal memuat `id`, `fruit`, `stage`, `quality`,
`confidence`, `box`, `mask` bila tersedia, `days_range` bila terkalibrasi,
`issues`, dan `explanation`.

### Frontend React

- React + TypeScript + Vite agar ringan dan mudah dijalankan di Windows.
- `getUserMedia` untuk kamera browser pada `localhost` atau HTTPS.
- Mode kamera snapshot dan mode unggah gambar.
- Canvas overlay untuk kotak/mask, nomor objek, dan warna status.
- Panel hasil per objek, panel ringkasan, panel sumber keputusan, dan panel
  peringatan confidence rendah.
- Tombol "ambil foto", "unggah", "tampilkan mask", dan "unduh laporan JSON".
- Aplikasi tidak mengklaim penyakit pasti. Label tampilan menggunakan
  "indikasi visual" atau "kemungkinan masalah".

### Rancangan halaman

```text
┌──────────────────────────────────────────────────────────────────┐
│ FruitVision ID        Kamera | Unggah | Dataset | Evaluasi       │
├───────────────────────────────┬──────────────────────────────────┤
│                               │ Ringkasan                         │
│       pratinjau kamera        │ 7 objek · 3 matang · 1 bermasalah│
│   mask/kotak dan nomor buah   │                                  │
│                               │ [01] Mangga · siap · 2-4 hari    │
│ [Ambil foto] [Tampilkan mask] │ [02] Pisang · mentah · 3-5 hari  │
│                               │ [03] Jeruk · bercak teramati    │
├───────────────────────────────┴──────────────────────────────────┤
│ Penjelasan: warna dan tekstur terlihat ...                       │
│ Catatan: hasil foto, bukan diagnosis penyakit pasti.             │
└──────────────────────────────────────────────────────────────────┘
```

Visual memakai tema "meja sortir pasar tropis": dasar gelap hijau daun,
warna aksen kuning pisang dan merah matang, dengan warna status yang tetap
memiliki kontras tinggi. Satu area kamera menjadi fokus, panel informasi tetap
tenang agar hasil tidak tertutup dekorasi.

## 8. Loop pencarian model terbaik

Loop harus bisa diulang dari command line dan menyimpan hasil ke JSON/CSV:

1. Audit manifest dataset: lisensi, jumlah gambar, kelas, ukuran, format,
   label kosong, label di luar gambar, duplikasi perceptual hash, dan sumber.
2. Normalisasi nama kelas ke skema internal tanpa menghapus label asli.
3. Buat split berdasarkan sumber dan `fruit_id`.
4. Latih baseline klasifikasi dari crop.
5. Latih detector bounding box.
6. Latih atau uji instance segmentation pada subset objek berdempetan.
7. Uji classifier kematangan dan kualitas di atas crop hasil model, bukan hanya
   crop bersih.
8. Uji konfigurasi augmentation yang relevan: perubahan cahaya, blur ringan,
   rotasi, occlusion, dan background. Hindari augmentasi yang mengubah makna
   warna kematangan secara berlebihan.
9. Ukur metrik pada test tetap dan subset khusus: objek kecil, berdempetan,
   occlusion, cahaya redup, dan buah yang tidak dikenal.
10. Beri peringkat dengan bobot: recall deteksi 30%, recall objek berdempetan
    20%, macro-F1 kematangan 20%, macro-F1 kualitas 15%, kalibrasi 10%, dan
    latency CPU 5%.
11. Simpan model terbaik, model runner-up, confusion matrix, contoh false
    positive/negative, dan alasan pemilihan.
12. Jangan memilih model hanya dari mAP tinggi jika recall pada objek
    berdempetan atau kelas lokal buruk.

Target awal, bukan janji hasil: recall objek berdempetan minimal 0,80 pada
subset lokal, macro-F1 tahap minimal 0,75, interval hari memiliki coverage
terukur, dan snapshot CPU terasa cukup cepat. Nilai final harus berasal dari
hasil pengujian.

## 9. Metrik dan pengujian

- Deteksi/segmentasi: precision, recall, mAP50, mAP50-95, per-class recall,
  crowded-subset recall, dan jumlah objek terhitung.
- Kematangan/kualitas: macro-F1, balanced accuracy, confusion matrix,
  calibration error, serta hasil per jenis buah.
- Estimasi hari: MAE, median absolute error, coverage interval 80% dan 90%,
  serta proporsi kasus yang ditolak karena confidence rendah.
- Web: `GET /health`, response schema, gambar rusak, gambar tanpa buah, banyak
  buah, objek berdempetan, kamera ditolak, dan backend tidak aktif.
- Uji publik antar-sumber: satu sumber untuk training dan sumber berbeda untuk
  test bila label serta lisensinya memungkinkan. Jika gambar berasal dari seri
  atau sesi yang sama, seluruh kelompok tetap berada pada satu split.
- Uji kamera dasar: gunakan file gambar publik yang sudah tersedia atau gambar
  yang ditampilkan di layar. Ini hanya smoke test antarmuka kamera, bukan
  validasi kondisi lapangan.
- Uji tanpa buah, objek bukan buah, buah berdempetan, buah di luar kelas, dan
  pencahayaan berbeda memakai koleksi publik atau variasi yang dapat diaudit.

Karena tidak ada foto kamera lokal yang dikumpulkan sendiri, laporan wajib
memiliki bagian "keterbatasan domain". Hasil test publik belum mengukur
perbedaan kamera, latar, pencahayaan, dan kebiasaan penataan buah di lingkungan
pengguna.

## 10. Struktur proyek yang diusulkan

```text
FruitVision_ID/
├─ backend/
│  ├─ app/
│  │  ├─ api/
│  │  ├─ schemas/
│  │  ├─ services/
│  │  ├─ inference/
│  │  └─ main.py
│  ├─ tests/
│  └─ requirements.txt
├─ frontend/
│  ├─ src/
│  │  ├─ components/
│  │  ├─ features/camera/
│  │  ├─ features/results/
│  │  └─ lib/
│  └─ package.json
├─ data/
│  ├─ manifests/
│  ├─ raw/               # tidak dikomit dan tidak dihapus otomatis
│  ├─ processed/
│  └─ local_longitudinal/
├─ models/
├─ research/
│  ├─ dataset_registry.yaml
│  ├─ audit_dataset.py
│  ├─ benchmark_loop.py
│  └─ reports/
├─ docs/
│  └─ superpowers/specs/
└─ README.md
```

## 11. Keamanan dan lisensi

- Tidak menyimpan foto kamera permanen tanpa pilihan pengguna.
- API lokal bind ke localhost secara default.
- Tidak menaruh token atau kredensial di source code.
- Setiap dataset memiliki field URL, tanggal akses, lisensi, checksum, status
  izin komersial, dan catatan turunan.
- Dataset tanpa lisensi tidak dipakai dalam model yang dibagikan.
- Lisensi Ultralytics dan setiap bobot pretrained harus dicatat sebelum
  distribusi model.

## 12. Keputusan yang sudah ditetapkan pengguna

- Tahap pertama adalah MVP terlatih, bukan kerangka model dummy.
- Distribusi untuk tugas pribadi/kelas, bukan produk komersial.
- Dataset diunduh bertahap dalam bentuk subset yang sudah diaudit.
- Keluaran wajib MVP: deteksi jenis, jumlah, posisi, buah berdempetan, dan
  kematangan.
- Bounding box wajib, mask opsional.
- Kualitas, penyakit, dan masalah visual ditunda dari MVP.
- Input di luar kelas ditolak dengan peringatan confidence rendah.
- Target inferensi adalah snapshot maksimal lima detik pada laptop lokal.
- Aplikasi harus berjalan offline setelah pemasangan.
- Tidak perlu mengambil foto baru. Foto uji berasal dari dataset publik atau
  repositori internet yang metadata dan lisensinya dicatat.

## 13. Keputusan builder yang masih harus diambil

- Apakah training dilakukan lokal atau di Google Colab. Desain mendukung
  keduanya, tetapi laptop lebih cocok untuk audit dan inferensi.
- Jika salah satu dari lima buah tidak memiliki label kematangan yang cukup,
  builder akan mempertahankan deteksi jenis dan menonaktifkan kematangan untuk
  jenis tersebut, dengan peringatan di UI.
- Bobot benchmark boleh diubah setelah hasil uji pertama jika kebutuhan
  kuliah lebih menekankan pemisahan objek daripada latency.

## 14. Rencana tahap setelah desain disetujui

1. Membuat manifest dan skrip audit dataset tanpa mengunduh semua arsip besar.
2. Menguji lisensi, struktur label, checksum, duplikasi, dan contoh visual.
3. Membuat baseline backend dengan model dummy yang eksplisit diberi label
   `demo`, lalu menggantinya dengan model terlatih setelah data siap.
4. Membuat frontend kamera dan unggah yang dapat diuji terhadap response dummy.
5. Menyiapkan notebook/skrip training, benchmark loop, dan laporan metrik.
6. Mengganti model dummy dengan bobot hasil benchmark dan menguji pada gambar
   publik yang belum pernah masuk training, sambil mencatat keterbatasan bahwa
   validasi kamera lokal baru berupa smoke test.

Tidak ada klaim akurasi atau estimasi hari sebelum tahap evaluasi menghasilkan
angka yang dapat diperiksa.
