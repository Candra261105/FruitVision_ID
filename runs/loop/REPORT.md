# Laporan Loop FruitVision

Dibuat: 2026-11-19 · sumber data: `runs/loop/iterations.jsonl`, audit terbaru `runs/loop/dataset-audits/`

## Status model

| Model | Status gerbang | Metrik kunci | Catatan |
|---|---|---|---|
| detector | gagal | test tiga kelas mAP50 0.528 · AP50 cucumber 0.548 · tomato 0.568 · green beans 0.469 | `cc0-fixed`, dataset eksperimen bersih, gate mAP50 >= 0.55 belum tercapai |
| maturity | eligible | macro-F1 0.782 · support 721 · balanced 0.800 |  |
| rot | eligible | macro-F1 0.755 · support 768 · balanced 0.913 |  |
| grade | eligible | macro-F1 0.835 · support 76 · balanced 0.830 |  |
| disease | eligible | macro-F1 1.000 · support 434 · balanced 1.000 |  |

## Riwayat iterasi

| Target | Iterasi | Tahap | Hasil | Keterangan |
|---|---|---|---|---|
| maturity | 1 | train | LULUS | training maturity completed |
| maturity | 2 | train | LULUS | gate eligible macro-F1 0.782 |
| maturity | 3 | registry | LULUS | registry runtime eligible |
| rot | 1 | train | LULUS | training rot completed |
| rot | 2 | train | LULUS | gate eligible macro-F1 0.755 |
| rot | 3 | registry | LULUS | registry runtime eligible |
| disease | 1 | train | LULUS | training disease completed |
| disease | 2 | train | LULUS | gate eligible macro-F1 1.000 |
| disease | 3 | registry | LULUS | registry runtime eligible |
| grade | 1 | train | LULUS | training grade completed |
| grade | 2 | train | LULUS | gate eligible macro-F1 0.835 |
| grade | 3 | registry | LULUS | registry runtime eligible |

## Kegagalan dan tindakan

- Detector `detector-512` selesai training, tetapi gagal gate pada test holdout. Metrik: mAP50 0.534842, mAP50-95 0.286666, precision 0.525587, recall 0.558107.
- Evaluasi ulang bobot `runs/detect/runs/loop-detector-cc0-fixed/experiment-512-fixed/weights/best.pt` pada test tiga kelas menghasilkan mAP50 0.528179, mAP50-95 0.272403, precision 0.517656, recall 0.548507. AP50: cucumber 0.547651, tomato 0.568055, green beans 0.468831. Gate tetap gagal karena mAP50 < 0.55.
- Dataset tambahan `LFTTim/multiple_smartphone_bangladesh_vegetable_detection` tidak dipakai untuk retraining. Audit `research/smartphone_cross_source.audit.json` menemukan 3.174/3.174 gambar target identik secara SHA-256 dengan `combined-top3-final`, termasuk benturan split lintas sumber. Label identik, sehingga menambahkannya hanya menggandakan data dan berisiko leakage.
- Dataset `combined-top3-cc0-experiment` diaudit bersih: 4.086 gambar, train 3.268, val 319, test 320, external holdout 179, tanpa hash duplikat lintas split dan tanpa label invalid.
- `kitchen-top3` memiliki 120 gambar unik dan label valid, tetapi hanya berisi cucumber/tomato (82 box). Evaluasi eksternal menghasilkan mAP50 0.229693, AP50 cucumber 0.001841 dan tomato 0.457545. Ini bukan holdout tiga kelas dan tidak dipakai sebagai gate promosi.
- `lvis-top3` seluruh 17 gambarnya duplikat dataset eksperimen. `smartphone-top3` seluruh 3.174 gambarnya juga duplikat. Keduanya tidak dipakai sebagai bukti eksternal.
- Training YOLO26m sedang berjalan pada output terpisah `runs/detect/runs/loop-detector-yolo26m/experiment-640-yolo26m`. Per 2026-11-20 12:25 log baru mencapai epoch 1, batch 143/1.634 pada CPU. `best.pt` belum tersedia dan bobot belum dievaluasi atau dipasang.
- Training YOLO26m tersebut dihentikan atas permintaan user. Eksperimen pengganti YOLO26n berjalan di `runs/detect/runs/loop-detector-yolo26n/experiment-512-yolo26n` dengan 30 epoch, `imgsz=512`, 8 CPU threads, dan batch 2. PyTorch terpasang sebagai CPU-only (`torch.cuda.is_available() == False`), sedangkan Vega 8 Windows tidak memiliki backend GPU training yang didukung Ultralytics. Log awal terverifikasi memakai `device=cpu`, bukan GPU.

## Audit kandidat dataset eksternal

- `henningheyen/Fruits-And-Vegetables-Detection-Dataset` ditahan, belum diunduh atau digabungkan. README menyatakan subset LVIS yang dikonversi ke YOLO, dengan 8.221 gambar, 63 kelas, 6.721 train, 1.500 validation, dan 180 test manual.
- Tiga target tersedia dengan mapping eksplisit: `cucumber/cuke`, `green bean`, serta dua entri tomato (`Tomato` dan `tomato`). Remapping tidak boleh dilakukan sebelum seluruh label dan split diaudit.
- `LVIS_Fruits_And_Vegetables/data.yaml` tidak konsisten dengan README: komentar split menyebut `train` 1.000 dan `val` 7.221, sementara `val` menunjuk ke `images/test` dan `test` juga menunjuk ke `images/test`. Ini cukup untuk menahan kandidat sampai struktur sebenarnya diverifikasi dari artefak unduhan.
- LICENSE repo adalah MIT dan README memang menyebut sumber LVIS, tetapi tidak memberikan lisensi atau pernyataan hak redistribusi untuk gambar LVIS. Lisensi kode LVIS yang ditemukan adalah BSD 3-Clause, bukan bukti lisensi gambar. Karena provenance hak gambar belum jelas, kandidat tidak memenuhi gate lisensi.
- Mirror Hugging Face `henningheyen/LVIS_Fruits_And_Vegetables` berlabel Apache-2.0, tetapi metadata yang tersedia hanya memiliki fitur `image` dan label split, tanpa bounding box. Mirror tersebut tidak dapat dipakai sebagai pengganti anotasi detector.
- Keputusan: **HOLD/REJECT FOR NOW**. Tidak ada data eksternal yang masuk ke train, validation, test, atau runtime. Kandidat hanya dapat dipertimbangkan ulang bila sumber gambar, lisensi, jumlah label, struktur split, hash duplikat, dan remapping tiga kelas terverifikasi.

## Kesimpulan jujur

Model aktif (lolos gerbang): maturity, rot, grade, disease. Detector custom belum aktif. Runtime masih memakai `models/yolov8s-world.pt` sebagai fallback demo.

Status `eligible` hanya berarti metrik test holdout melewati ambang yang ditetapkan. Model tidak diklaim berlaku di luar domain dataset sumbernya.

## Perubahan runtime yang sudah diuji

- Fallback YOLO-World mencoba confidence review rendah ketika hasil utama kosong. Crop kecil dicoba dengan konteks padding, sedangkan gambar landscape besar dicoba ulang pada frame asli.
- Fallback landscape kosong kini mencoba tiga tile horizontal dengan pemetaan bbox ke koordinat gambar asli dan deduplikasi IoU.
- Confidence rendah ditampilkan sebagai `review`, bukan dibuang diam-diam.
- Bbox yang menyentuh tepi diberi warning. Bbox yang hampir memenuhi frame diberi warning kemungkinan crop/objek terpotong dan status `review`.
- Enhancement tetap bekerja pada salinan gambar. Gambar asli dikembalikan bersama gambar hasil enhancement.
- Backend test: 116 lulus. Frontend test: 7 lulus. Frontend production build: lulus.
- Verifikasi terbaru setelah tiling: backend test 117 lulus, frontend test 7 lulus, frontend build lulus. Smoke API 14 gambar: 14/14 HTTP 200 dan 7/14 menghasilkan deteksi. Ini uji fungsi aplikasi, bukan metrik akurasi.

