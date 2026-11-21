# Riset dataset dan training

`dataset_registry.yaml` adalah daftar kandidat sumber. Status `metadata-only`
berarti metadata sudah dicatat, bukan berarti dataset sudah lolos audit.

Tidak ada script di folder ini yang mengunduh dataset otomatis. Alur aman:

```powershell
& ..\.venv\Scripts\python.exe research\audit_dataset.py --manifest research\dataset_registry.yaml
& ..\.venv\Scripts\python.exe research\audit_dataset.py --manifest research\dataset_registry.yaml --source D:\dataset-yang-sudah-diunduh
& ..\.venv\Scripts\python.exe research\audit_dataset.py --manifest research\dataset_registry.yaml --source D:\lvis --source D:\bangladesh --source D:\kitchen --output research\reports\combined-audit.json
& ..\.venv\Scripts\python.exe research\prepare_yolo.py --source D:\dataset-yolo --output research\reports\label-audit.json
```

Audit lokal membaca format COCO dan YOLO, memeriksa box yang rusak, menormalkan
varian nama kelas yang jelas, dan menghitung SHA-256 gambar. Jika `--source`
diulang, laporan juga membuat ranking Top 3 berdasarkan box valid gabungan.
Gambar dengan hash sama hanya dihitung sekali dalam ranking gabungan. Namun
jumlah box tetap mengikuti anotasi terbaik per gambar dan kelas, sehingga
perbedaan anotasi lintas sumber tidak dijumlahkan dua kali.

Perintah gabungan di atas belum diuji pada dataset nyata di mesin ini karena
folder `D:\lvis`, `D:\bangladesh`, dan `D:\kitchen` belum tersedia. Jalankan
setelah dataset diunduh secara eksplisit dan periksa `aggregate.top3` sebelum
menyiapkan data training.

Training membutuhkan `ultralytics` dan file `data.yaml` yang sudah diperiksa.
Script training tidak dijalankan otomatis karena bobot pretrained dan dataset
bisa berukuran besar.

Jika data tidak memiliki label untuk semua lima buah, jangan mengisi label
yang hilang dengan tebakan. API akan tetap menjaga status kelas dan kematangan
secara eksplisit.
