# Desain training ulang FruitVision dengan YOLO26m

## Tujuan

Melatih detector custom FruitVision dengan YOLO26m untuk mendeteksi tiga kelas:
`cucumber`, `tomato`, dan `green beans`. Model baru hanya boleh menggantikan
fallback demo jika bukti evaluasinya memenuhi gate proyek.

## Pendekatan yang dipilih

YOLO26m pretrained dipakai sebagai titik awal fine-tuning pada dataset bersih
`data/processed/combined-top3-cc0-experiment/data.yaml`. YOLO11m tidak dilatih
karena pengguna memilih langsung YOLO26m. YOLO-World tetap menjadi fallback
selama model baru belum eligible.

## Evaluasi dan gate

Training memakai CPU dengan `workers=0`, seed tetap, validasi selama training,
dan output terpisah dari bobot aktif. Bobot terbaik dievaluasi ulang pada test
tiga kelas menggunakan evaluator final yang mencatat SHA-256 dan provenance.

Model hanya eligible jika semua syarat berikut terpenuhi:

- mAP50 keseluruhan minimal `0,55`
- AP50 `cucumber` minimal `0,40`
- AP50 `tomato` minimal `0,40`
- AP50 `green beans` minimal `0,40`

Jika bobot pretrained atau arsitektur YOLO26m tidak tersedia/kompatibel pada
versi Ultralytics lokal, proses berhenti dengan status jelas. Tidak ada fallback
diam-diam ke YOLO11 atau penggantian runtime otomatis.

## Batasan

- Laptop hanya memiliki CPU Ryzen 5 PRO 3500U, sehingga YOLO26m dapat memakan
  waktu lama dan batch harus disesuaikan dengan RAM.
- External holdout tiga kelas yang valid belum tersedia. Holdout `kitchen-top3`
  hanya memiliki cucumber dan tomato, sehingga tidak menjadi gate utama.
- Model yang gagal gate tetap disimpan sebagai artefak eksperimen, tetapi tidak
  disalin ke `models/detector.pt` dan tidak dipakai aplikasi.

## Hasil yang diharapkan

Artefak training, log, metrik test, hash bobot, dan keputusan gate tersimpan di
`runs/loop/`. Runtime tetap aman apabila hasil YOLO26m belum memenuhi syarat.
