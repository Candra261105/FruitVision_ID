# Tomato Pumpkin Lemon

Dataset eksperimen terpisah dengan urutan kelas `0 tomato`, `1 pumpkin`, `2 lemon`.

Semua gambar pada tahap ini berasal dari sumber beranotasi asli. Folder `external_holdout` dipakai sebagai evaluasi terpisah. Data sintetis, bila ditambahkan, hanya boleh masuk `train`.
`data.yaml` memakai `path: dataset` karena file konfigurasi berada di dalam
folder `dataset`, sedangkan Ultralytics menyelesaikan `path` relatif terhadap
folder paket saat konfigurasi dipanggil melalui trainer.
