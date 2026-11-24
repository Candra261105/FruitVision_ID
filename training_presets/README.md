# Preset training tomato/pumpkin/lemon

Klik dua kali salah satu file `train_*_epochs.cmd` dari komputer yang sudah
memiliki `.venv`, dependency backend, dan folder dataset.

Semua preset langsung menunjuk ke:

```text
data\processed\tomato-pumpkin-lemon\data.yaml
```

Urutan kelas tetap `0 tomato`, `1 pumpkin`, `2 lemon`. Script melakukan validasi
dataset lebih dulu. Training belum dijalankan oleh repository ini dan jumlah
epoch tidak menjamin akurasi.
