FruitVision YOLO26n checkpoint dijeda
Dibuat: 2026-11-20 11:31:55 +07:00
Run asli: C:\Users\dragon\Downloads\FruitVision_ID\runs\detect\runs\loop-detector-yolo26n\experiment-512-yolo26n
Checkpoint resume: C:\Users\dragon\Downloads\FruitVision_ID\runs\loop\checkpoints\yolo26n-paused-20261120-113155\last.pt
Checkpoint best sementara: C:\Users\dragon\Downloads\FruitVision_ID\runs\loop\checkpoints\yolo26n-paused-20261120-113155\best.pt
last.pt=8cc61b33a050cfe27bb807d4302ae599fb6df76996b541def4f1ee64eaa81297 best.pt=8efd7122ee191c2a9c1ae82b3edcab04601fd7871b06f7d21e34fcbdc42c3b2a
Baris results.csv terakhir: 19,5943.5,1.65418,1.68966,0.00925,0.44406,0.51472,0.45191,0.23961,1.50822,1.74106,0.01145,0.000580174,0.000580174,0.000580174

Cara melanjutkan:
1. Jangan hapus atau pindahkan folder run asli.
2. Set environment variable FRUITVISION_RESUME_CHECKPOINT ke file last.pt di folder ini.
3. Jalankan runs\loop\train_yolo26m.py dengan Python .venv.
4. Training akan resume dari epoch tersimpan menuju target epoch 30.
5. Setelah selesai, evaluasi best.pt pada test utama dan external holdout. Jangan pasang ke runtime sebelum gate.

Catatan: last.pt dipakai untuk melanjutkan karena menyimpan state optimizer. best.pt adalah bobot terbaik sementara.
