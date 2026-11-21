# Desain Sequential Detector Fallback FruitVision

## Intent

Menambahkan alur detector utama + YOLO-World fallback agar FruitVision tetap mencoba detector custom yang sudah lolos gate, tetapi masih dapat memberi hasil review ketika detector utama kosong atau gagal dijalankan.

## Tujuan

- Detector custom hanya dipakai jika status `.gate.json` adalah `eligible`.
- YOLO-World tetap tersedia sebagai fallback demo.
- Hasil detector custom yang valid tidak ditimpa oleh fallback.
- Kegagalan model tidak menjatuhkan request aplikasi.
- Sumber setiap hasil deteksi dapat diaudit sebagai `custom` atau `world_fallback`.
- Analisis maturity, rot, disease, dan grade tetap memakai crop dari bounding box akhir.

## Bukan tujuan

- Menggabungkan bounding box dari dua model pada tahap ini.
- Mempromosikan model yang gagal gate.
- Mengubah validation/test utama atau mengklaim peningkatan akurasi sebelum evaluasi ulang.
- Mengubah model kondisi yang sudah ada.

## Pendekatan yang dipertimbangkan

1. **Sequential fallback, dipilih.** Jalankan custom lebih dulu, lalu YOLO-World hanya jika custom kosong atau gagal. Paling aman untuk CPU dan false positive.
2. **Ensemble langsung.** Jalankan dua model dan gabungkan box. Berpotensi menaikkan recall, tetapi lebih lambat dan dapat menambah false positive. Ditunda sampai baseline fallback diuji.
3. **Routing berdasarkan kondisi foto.** Memilih model dari brightness, blur, atau ukuran objek. Lebih rumit dan rawan overfit sebelum ada bukti per kondisi.

## Keputusan desain

- Registry memuat dua runtime: `primary` dan `fallback`.
- Runtime primary berasal dari path detector custom yang dikonfigurasi dan hanya boleh aktif bila gate eligible.
- Runtime fallback berasal dari `models/yolov8s-world.pt` dan memakai prompt tiga kelas target.
- Jika primary menghasilkan setidaknya satu kandidat valid, hanya hasil primary yang diteruskan. Confidence rendah tetap ditandai `review` oleh service yang sudah ada.
- Jika primary tidak menghasilkan kandidat, tidak tersedia, atau melempar error, fallback dijalankan.
- Jika fallback hanya menghasilkan kandidat confidence rendah, hasil tetap terlihat sebagai `review`, bukan dikonversi menjadi accepted.
- API menambahkan sumber detector pada setiap hasil agar perbedaan custom dan fallback terlihat.
- Jika tidak ada primary eligible, aplikasi tetap menggunakan YOLO-World sebagai fallback tunggal dengan status demo.

## Verifikasi

- Unit test untuk primary berhasil, primary kosong, primary error, fallback kosong, dan primary gate tidak eligible.
- Backend test penuh.
- Frontend test dan production build.
- Smoke API.
- Evaluasi mAP50/AP50 tetap memakai test dan external holdout yang tidak berubah.
- Model hanya dipromosikan jika gate `mAP50 >= 0,55` dan AP50 tiap kelas `>= 0,40` tetap terpenuhi.

## Keputusan yang masih dibutuhkan

Persetujuan pengguna atas desain sequential fallback ini sebelum implementasi kode.
