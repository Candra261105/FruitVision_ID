FruitVision YOLO26n checkpoint dijeda
Dibuat: 2026-11-20 08:53:48 +07:00
Run asli: C:\Users\dragon\Downloads\FruitVision_ID\runs\detect\runs\loop-detector-yolo26n\experiment-512-yolo26n
Checkpoint resume: C:\Users\dragon\Downloads\FruitVision_ID\runs\loop\checkpoints\yolo26n-paused-20261120-085300\last.pt
Checkpoint best sementara: C:\Users\dragon\Downloads\FruitVision_ID\runs\loop\checkpoints\yolo26n-paused-20261120-085300\best.pt
SHA256 last.pt: a9ddbc304bb4a9d720aff9ddfed7faa3015ba3f80c9c98fef4aaffdfe790faea
SHA256 best.pt: 92342f59149e42d2378c4582704e4b440418a82eb680433c3c6cebfd2ed483fe

Cara melanjutkan:
1. Jangan hapus atau pindahkan folder run asli.
2. Jalankan training dengan Ultralytics resume dari file last.pt.
3. Target tetap epoch 30, sehingga checkpoint melanjutkan dari epoch terakhir yang tersimpan, bukan mulai dari nol.
4. Setelah selesai, evaluasi weights/best.pt pada test utama dan external holdout. Jangan pasang ke runtime sebelum gate.

Catatan: best.pt adalah bobot terbaik sementara. last.pt dipakai untuk melanjutkan training karena menyimpan state optimizer.
