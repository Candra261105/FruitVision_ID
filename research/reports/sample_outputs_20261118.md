# Contoh output pipeline per jenis input

Mode detector: `demo` · threshold keyakinan 0.45

| Kasus | Input | Objek terdeteksi | Pesan | Enhancement |
|---|---|---|---|---|
| tomato | `IMG_004.jpg` | - | Tidak ada objek dari tiga kelas yang dikenali. | enhanced |
| cucumber | `IMG_003.jpg` | Mentimun 51% | 1 objek terdeteksi. 1 objek dinilai kondisi mentimun. | enhanced |
| green_beans | `bangladesh_000003.jpg` | - | Tidak ada objek dari tiga kelas yang dikenali. | enhanced |
| no_fruit | `IMG_001.jpg` | - | Tidak ada objek dari tiga kelas yang dikenali. | enhanced |
| dark_tomato | `IMG_004.jpg` | - | Tidak ada objek dari tiga kelas yang dikenali. | enhanced |
| blurry_cucumber | `IMG_003.jpg` | Mentimun 51% | 1 objek terdeteksi. 1 objek dinilai kondisi mentimun. | enhanced |

## Status fitur lanjutan saat pengujian

| Fitur | Status |
|---|---|
| detector | available |
| maturity | available |
| rot | available |
| disease | available |
| grade | available |
