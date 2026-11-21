# Desain Perbaikan Terarah YOLO26n

## Intent

Meningkatkan kandidat YOLO26n yang sudah dilatih sampai epoch 30 tanpa memulai dari bobot awal, lalu memasangnya ke FruitVision hanya jika evaluasi ulang memenuhi gate model.

Baseline saat ini:

- Bobot: `runs/detect/runs/loop-detector-yolo26n/experiment-512-yolo26n/weights/best.pt`
- Dataset: `data/processed/combined-top3-cc0-experiment`
- Test utama: 320 gambar
- mAP50 test: `0,5141659247`
- AP50 cucumber: `0,5152481404`
- AP50 tomato: `0,5820062301`
- AP50 green beans: `0,4452434036`
- Gate: rejected karena mAP50 belum mencapai `0,55`

## Tujuan

- Mempertahankan YOLO26n sebagai detector custom.
- Menemukan sumber error terbesar sebelum training lanjutan.
- Memperbaiki train split secara terarah tanpa mengubah validation, test, atau external holdout.
- Melanjutkan dari checkpoint epoch 30, bukan mulai dari `yolo26n.pt`.
- Memasang model custom hanya jika gate terpenuhi.

## Bukan tujuan

- Mengganti YOLO26n dengan model lain.
- Mengulang training identik tanpa analisis.
- Memasukkan test utama atau external holdout ke train.
- Menambahkan dataset eksternal baru pada putaran ini.
- Mengklaim peningkatan sebelum evaluasi ulang.
- Memasang model yang masih berstatus `rejected`.

## Pendekatan yang dipertimbangkan

1. **Lanjut epoch langsung dengan data lama.** Cepat, tetapi tidak menjawab sumber error dan berisiko stagnan. Tidak dipilih.
2. **Analisis error lalu perbaikan train dan augmentasi terarah.** Dipilih karena mAP50 hanya kurang `0,0358`, tetap menjaga test yang sama, dan memberi perubahan yang dapat diaudit.
3. **Menambah dataset eksternal baru.** Ditunda karena perlu audit provenance, lisensi, duplikat, kelas, dan distribusi data tambahan.

## Rencana kerja

### Fase 1: Analisis baseline

- Jalankan prediksi `best.pt` pada seluruh test utama.
- Cocokkan prediksi dengan label YOLO.
- Buat laporan per kelas dan per gambar yang mencakup:
  - false negative
  - false positive
  - salah kelas
  - IoU rendah
  - objek kecil atau dekat tepi
  - confidence dan ukuran box
- Buat ringkasan kelas yang paling bermasalah.

### Fase 2: Audit dan perbaikan train

- Periksa sampel train yang memiliki pola error serupa.
- Perbaiki hanya label train yang terbukti salah atau tidak konsisten.
- Jangan menyalin gambar test/validation/external holdout ke train.
- Jika audit tidak menemukan label rusak, gunakan augmentasi terarah yang masih tersedia di konfigurasi Ultralytics, terutama untuk skala, crop ringan, dan variasi pencahayaan.
- Simpan dataset hasil perbaikan sebagai eksperimen baru, bukan menimpa dataset baseline.
- Tulis manifest perubahan dan jumlah file yang berubah.

### Fase 3: Training lanjutan

- Gunakan `best.pt` epoch 30 sebagai bobot awal.
- Simpan run pada folder baru agar baseline tidak tertimpa.
- Gunakan validation yang sama dan test tetap tidak dipakai selama training.
- Hentikan bila proses tidak stabil atau artefak tidak valid.

### Fase 4: Evaluasi dan keputusan

- Evaluasi kandidat pada test utama 320 gambar dengan script evaluasi resmi.
- Evaluasi external holdout 179 gambar secara terpisah jika data YAML/manifest holdout tersedia.
- Bandingkan dengan baseline mAP50 `0,5142` dan AP50 tiap kelas.
- Kandidat eligible hanya bila:
  - mAP50 keseluruhan `>= 0,55`
  - AP50 cucumber `>= 0,40`
  - AP50 tomato `>= 0,40`
  - AP50 green beans `>= 0,40`
- Jika gagal, simpan sebagai kandidat rejected dan runtime tetap memakai YOLO-World.

### Fase 5: Pemasangan aman

- Salin hanya kandidat eligible ke `models/` dengan nama model yang jelas.
- Salin `.gate.json` bersama bobot.
- Ubah konfigurasi runtime hanya setelah gate eligible.
- Jalankan test backend, build frontend, dan smoke API.
- Jika kandidat gagal, tidak ada perubahan runtime.

## Kriteria selesai

Pekerjaan dianggap selesai bila analisis baseline, manifest perbaikan, training lanjutan, evaluasi test, evaluasi holdout, dan keputusan deployment terdokumentasi. "Selesai" tidak berarti model pasti dipasang. Pemasangan bergantung pada gate hasil evaluasi.

## Keputusan yang sudah ditetapkan

- YOLO26n dipertahankan.
- Training lanjutan harus berangkat dari bobot epoch 30.
- Test utama tetap sama.
- Dataset eksternal baru tidak ditambahkan pada putaran ini.
- Model gagal gate tidak boleh menggantikan YOLO-World.
