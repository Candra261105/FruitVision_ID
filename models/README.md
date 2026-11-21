# Model kondisi FruitVision ID

Folder ini menyimpan bobot detector dan model kondisi lanjutan. Backend hanya memakai
model kondisi jika ada pasangan:

```text
models/<task>.pt
models/<task>.gate.json
```

dengan `status: "eligible"`, provenance lengkap, hash bobot cocok, dan evaluasi yang
memenuhi ambang gate.

## Isi saat ini

| File | Task | Kelas | Cakupan |
|---|---|---|---|
| `yolov8s-world.pt` | detector | cucumber, tomato, green beans | fallback demo, bukan model custom |
| `maturity.pt` + `maturity.gate.json` | kematangan | `unripe`, `half_ripe`, `ripe` | tomat |
| `rot.pt` + `rot.gate.json` | busuk | `not_rotten`, `rotten` | tomat |

`disease` dan `grade` sengaja belum punya bobot karena belum ada dataset berlabel
yang layak.

## Aturan gate

Model hanya boleh aktif jika:

- `macro_f1 >= 0.75`
- `balanced_accuracy >= 0.75`
- support test minimal 20 objek dan tiap kelas minimal 5
- `model_sha256` cocok dengan file bobot
- `taxonomy_sha256` cocok dengan `research/advanced_taxonomy.yaml`
- label evaluasi harus subset dari label taxonomy task tersebut (minimal 2 kelas),
  sehingga model yang hanya mengenali sebagian kelas tetap bisa dinilai secara jujur

Gate ditulis oleh `research/evaluate_advanced_classifier.py`. Jangan mengedit gate
manual. Jika taxonomy berubah, evaluasi harus dijalankan ulang.

## Batasan

Model kematangan dan busuk dilatih dari crop tomat 160x160 dataset ParsaVictor
(CC BY 4.0). Test holdout seluruhnya berasal dari sumber Tomato Defects, sehingga
angka evaluasi tidak otomatis berlaku untuk semua foto ponsel, pencahayaan, atau
latar belakang. Model tidak digunakan untuk komoditas lain.
