# Desain Ekspansi Dataset & Pemaksimalan Akurasi Deteksi Tomato, Pumpkin, Lemon

## 1. Konteks & Mengapa (Why)

Hasil pelatihan 218 epoch model sebelumnya mencapai akurasi keseluruhan **73.25% mAP50**:
- **Pumpkin**: **95.93%** (Kategori sangat matang, bentuk dan ukuran khas).
- **Tomato**: **69.10%** (Cukup baik, tetapi tomat hijau/kecil dan bertumpuk sering gagal deteksi).
- **Lemon**: **54.72%** (Bottleneck utama yang menarik turun performa keseluruhan).

Kurva `val/loss` pada training sebelumnya mendatar sejak epoch 100–140, membuktikan bahwa penambahan epoch pada dataset lama tidak lagi menaikkan akurasi karena variasi lemon dan tomat sudah jenuh. Untuk mendorong akurasi ke **85% – 95%+**, diperlukan ekspansi dataset masif melalui penggabungan sumber data nyata baru, pemanenan crop mandiri, dan komposit sintetis terkontrol.

---

## 2. Sasaran (Goals), Bukan Sasaran (Non-Goals), & Anti-Goals

### Goals
1. **Peningkatan Akurasi Lemon & Tomato**: Meningkatkan mAP50 Lemon dari ~54% ke >80% dan Tomato dari ~69% ke >85%, dengan mempertahankan Pumpkin di >95%.
2. **Koleksi Dataset Multi-Sumber**: Mengintegrasikan gambar nyata lemon dan tomat dari sumber terbuka terverifikasi (Roboflow Universe Lemon/Tomato, LVIS, OpenImages) yang bebas bias satu wilayah.
3. **Pemanenan Crop & Sintesis Komposit Mandiri**: Mengekstrak ribuan crop objek nyata (lemon, tomat, labu), lalu menghasilkan komposit multi-objek dengan variasi oklusi (menempel, bertumpuk), pencahayaan, skala (skala kecil/jauh), dan latar belakang meja/dapur.
4. **Isolasi Split Ketat (Leakage-Free)**: Memastikan data sintetis **hanya** masuk ke split `train`. Split `val` dan `test` **wajib 100% gambar nyata** tanpa percampuran hash/phash dengan data latih.
5. **Paket Pelatihan Colab Terakselerasi**: Menyediakan arsip `.zip` dan skrip satu-klik untuk Google Colab T4 GPU yang mendukung resolusi `imgsz=800` dan model `yolo11s.pt` (Small) untuk mendeteksi buah berukuran kecil.

### Non-Goals
- Mengubah skema kelas: Taksonomi tetap baku dan wajib `0: tomato`, `1: pumpkin`, `2: lemon`.
- Mengubah kontrak REST API backend (`/api/detect`) atau struktur respons JSON yang sudah dipakai frontend.

### Anti-Goals
- Melabeli buah lain (seperti jeruk mandarin, apel, atau mentimun) sebagai lemon atau tomat demi menambah kuantitas angka.
- Memasukkan gambar komposit sintetis ke dalam data validasi (`val`) atau pengujian (`test`). Hal ini akan menghasilkan evaluasi palsu (*synthetic overfitting*).

---

## 3. Pendekatan yang Dipertimbangkan & Keputusan Desain Terpilih

1. **Pendekatan 1: Penambahan Data Nyata Saja**
   Mengunduh gambar dari internet dan menambahkan bounding box. Baik untuk variasi visual, tetapi kurang memiliki variasi buah berhimpitan/bertumpuk di atas meja dapur.
2. **Pendekatan 2: Komposit Sintetis dari Data Lama Saja**
   Hanya melakukan crop dari 3.000 gambar Bangladesh lama. Lemah karena lemon di dataset Bangladesh mayoritas lemon hijau kecil, sehingga model tetap buta terhadap lemon kuning cerah atau lemon di supermarket.
3. **Pendekatan 3: Strategi Terpadu (Dipilih & Disetujui)**
   Menggabungkan dataset nyata eksternal, pemanenan crop mandiri terotomatisasi dari ribuan objek terverifikasi yang sudah ada di proyek (`cls_tomato` kematangan/busuk, dataset dapur `kitchen`, bounding box Bangladesh), menyusun komposit oklusi realistis di `train`, serta meng-upgrade konfigurasi pelatihan ke `yolo11s` pada resolusi 800px di Colab T4.

### Keputusan Kunci Hasil Builder Check:
1. **Otomasi Penuh Tanpa Tugas Manual**: User tidak perlu mengambil foto manual. Agen memanen crop secara terotomatisasi dari ribuan objek beresolusi tinggi yang sudah ada di proyek (`data/advanced/source/crops/cls_tomato/` dengan variasi raw/half_ripe/ripe/rotten, dataset dapur, dan bounding box Bangladesh).
2. **Benchmark Terkunci (Frozen Val & Test)**: Split `val` (300 gambar) dan `test` (300 gambar) dibekukan menggunakan set benchmark yang sama persis dengan pengujian sebelumnya. Ini menjamin perbandingan peningkatan akurasi (dari baseline 73.25%) bersifat 100% jujur, transparan, dan *apple-to-apple*.
3. **Target Komposit & Pembobotan Kelas**: Menghasilkan **1.500 gambar komposit baru** pada resolusi 800x800. Alokasi objek: **45% lemon**, **35% tomato**, dan **20% pumpkin** untuk memprioritaskan peningkatan akurasi kelas lemon dan tomat yang tertinggal.
4. **Latar Belakang Prosedural Cerdas**: Menggunakan generator latar belakang berbasis NumPy & Pillow (tekstur serat kayu, marmer dapur, gradien pencahayaan natural) sehingga berjalan instan, 100% offline, tanpa kanvas abu-abu polos.

---

## 4. Arsitektur & Alur Pipeline

```
[Sumber Data Bangladesh (Lama)] ──┐
[Dataset Terverifikasi (Baru)]  ──┼─> [Audit & Filter] ─> [Pemisahan Split Nyata (Val & Test Terisolasi)]
[Foto Mandiri/Lokal]            ──┘         │
                                            ▼
                                   [Crop Harvester]
                                  (Ribuan Objek Buah)
                                            │
                                            ▼
                               [Advanced Composite Engine]
                             (Sintetis Multi-Objek untuk Train)
                                            │
                                            ▼
                         [Dataset Gabungan: images/ & labels/]
                                  (Train, Val, Test)
                                            │
                                            ▼
                               [tomato-pumpkin-lemon-expanded.zip]
                                            │
                                            ▼
                                [Google Colab T4 GPU Training]
                                (Batch 32, imgsz 800, YOLO11s)
```

---

## 5. Rincian Modul yang Akan Dibangun

### A. Modul Pemanen Crop (`harvest_fruit_crops.py`)
- Membaca bounding box terverifikasi dari dataset nyata.
- Memotong objek dengan sedikit bantalan tepi (*padding* 4-8px) agar kontur buah utuh.
- Menyimpan metadata crop: kelas, dimensi asli, rasio aspek, dan tingkat ketajaman.

### B. Generator Komposit Lanjutan (`generate_fruit_composites.py`)
- Menghasilkan gambar latih sintetis berukuran 800x800 atau 640x640.
- Skenario penempatan:
  * *Isolated*: Buah terpisah dengan jarak jelas.
  * *Clustered*: Buah berkumpul dalam wadah/piring.
  * *Occluded*: Buah saling menumpuk sebagian (5%–35% tumpang tindih).
  * *Scale Jitter*: Ukuran bervariasi dari 24x24 px (jarak jauh) hingga 350x350 px (jarak dekat).
- Latar belakang: Warna netral, tekstur meja kayu, marmer dapur, dan kain.
- *Feathering*: Tepi crop diberi transparansi mikro (Gaussian edge blur 1px) agar potongan tidak membentuk tepi tajam buatan.

### C. Skrip Penggabung Dataset (`build_expanded_dataset.py`)
- Menggabungkan data asli + data baru + komposit sintetis ke folder `data/processed/tomato-pumpkin-lemon-expanded/`.
- Memastikan rasio per kelas seimbang (kuota lemon dinaikkan agar tidak kalah dominan dari tomat).
- Menghasilkan file manifes, audit, dan `data.yaml`.

### D. Skrip Pelatihan Colab Terpadu (`colab_train_expanded.py`)
- Menjalankan training dengan opsi model:
  * Model `yolo11n.pt` (cepat, 2.5M params) atau `yolo11s.pt` (presisi tinggi, 9.4M params).
  * Resolusi input `imgsz=800` (menangkap detail lemon dan tomat kecil).
  * Otomatis evaluasi, plot grafik kurva, dan *auto-download* bobot terbaik `best.pt`.

---

## 6. Verifikasi & Pengujian
- **Verifikasi Integritas Label**: 0 bounding box yang keluar dari batas 0.0–1.0, 0 bounding box dengan luas 0, dan kelas strictly {0, 1, 2}.
- **Verifikasi Bebas Kebocoran**: Tidak ada pHash yang sama antara `train` dan `val/test`.
- **Smoke Test Training**: Uji coba 1 epoch lokal untuk memvalidasi alur data sebelum di-zip ke Colab.
