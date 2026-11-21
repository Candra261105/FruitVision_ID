# Desain Training Penyakit, Kematangan, Busuk, dan Grade FruitVision ID

Tanggal: 2026-11-18  
Status: menunggu persetujuan sebelum implementasi

## Intent

Menyediakan empat model terpisah untuk objek `cucumber`, `tomato`, dan `green beans`:

- penyakit
- kematangan
- busuk
- grade kualitas

Hasil harus dapat diaudit dan tidak boleh tampil sebagai prediksi aplikasi sebelum model
lolos evaluasi serta gate provenance.

## Temuan audit saat ini

Dataset lokal yang tersedia hanya memiliki anotasi deteksi jenis objek. Tidak ditemukan
label khusus penyakit, tahap kematangan, busuk, atau grade. Kandidat eksternal di
`research/dataset_registry.yaml` masih berstatus metadata-only, dan sebagian besar
berisi pisang, mangga, atau jeruk, bukan tiga kelas target FruitVision ID.

Karena itu tidak ada model valid yang dapat dilatih dari data lokal saat ini. Melatih
fitur lanjutan dari nama kelas detector akan menghasilkan label palsu dan tidak boleh
dilakukan.

## Pendekatan yang dipertimbangkan

1. **Label valid per fitur, direkomendasikan**
   - Kumpulkan atau impor crop/gambar berlabel untuk tiga tanaman target.
   - Latih empat model classifier object-level secara terpisah.
   - Kelebihan: klaim sesuai label, evaluasi dapat dipisahkan, mudah menonaktifkan satu
     fitur bila datanya kurang. Kekurangan: membutuhkan anotasi tambahan.

2. **Pseudo-label dari aturan warna atau AI**
   - Membuat label otomatis lalu melatih model.
   - Tidak dipakai sebagai model produksi. Label buatan dapat memperkuat kesalahan dan
     tidak cukup untuk diagnosis penyakit atau keputusan grade.

3. **Transfer learning dari dataset buah lain**
   - Berguna sebagai eksperimen teknis, tetapi tidak menjadikan model valid untuk
     cucumber, tomato, atau green beans. Hasil hanya boleh berstatus research dan tidak
     diaktifkan di web.

Pendekatan yang dipilih: **pendekatan 1**, dengan scaffold loop yang aman untuk fitur
yang datanya belum ada. Pendekatan 2 dan 3 hanya boleh dijalankan jika diberi label
eksperimen dan tidak dipromosikan.

## Definisi label yang diusulkan

Label harus disetujui pemilik data atau standar sortir sebelum anotasi massal.

| Model | Label minimum | Catatan |
|---|---|---|
| Penyakit | `healthy`, penyakit-1, penyakit-2, ..., `unknown` | Nama penyakit harus berasal dari sumber/validator, bukan ditebak dari warna.
| Kematangan | `unripe`, `ripe`, `overripe`, `unknown` | Tidak diubah menjadi estimasi jumlah hari.
| Busuk | `not_rotten`, `rotten`, `unknown` | Pisahkan busuk dari luka mekanis, noda, dan penyakit bila standar membedakannya.
| Grade | `A`, `B`, `C`, `reject`, `unknown` | Arti grade harus ditulis dalam standar, termasuk toleransi cacat dan ukuran.

Label `unknown` dipakai untuk review atau dikeluarkan dari loss training, bukan dipaksa
menjadi kelas positif.

## Format data

Setiap record menyimpan minimal:

```json
{
  "image": "images/2026/session-001/img-0001.jpg",
  "object_id": "img-0001-obj-01",
  "fruit": "tomato",
  "bbox": [0.10, 0.12, 0.80, 0.76],
  "label": "ripe",
  "task": "maturity",
  "source": "operator-01",
  "session_id": "session-001",
  "annotator": "operator-01",
  "label_version": "v1",
  "license_or_permission": "internal",
  "split": "train"
}
```

Split dilakukan berdasarkan `session_id`, sumber, atau batch pengambilan, bukan secara
acak per frame. Foto berurutan dari sesi yang sama tidak boleh tersebar ke train dan
test karena akan membuat skor terlalu optimistis.

## Loop training

Implementasi akan menyediakan satu runner resumable dengan empat konfigurasi task:

```text
research/advanced_training_matrix.yaml
  disease
  maturity
  rot
  grade
```

Setiap task memiliki folder run sendiri, checkpoint `last.pt`/`best.pt`, log, metrics,
confusion matrix, manifest hash, konfigurasi, dan `status.json`. Runner harus:

- menghentikan training saat `STOP_TRAINING` muncul;
- melanjutkan dari checkpoint valid;
- menulis status JSON secara atomik;
- menandai `failed` jika terjadi exception;
- melewati task tanpa manifest berlabel dan menulis status `blocked_no_dataset`;
- tidak mengubah atau mengaktifkan model aplikasi otomatis.

Model awal yang dipilih adalah classifier ringan berbasis crop, bukan detector penyakit
yang langsung mengklaim diagnosis pada seluruh gambar. Inference lanjutan hanya berjalan
setelah detector menemukan objek dan model task tersebut berstatus `eligible`.

## Evaluasi dan gate

Setiap task dievaluasi pada test tetap yang tidak dipakai saat tuning. Laporan minimal:

- jumlah gambar dan objek per kelas;
- macro-F1, balanced accuracy, precision, recall per kelas;
- confusion matrix;
- calibration/error review untuk confidence;
- hasil per tanaman dan per sumber/session;
- contoh false positive dan false negative.

Model hanya diberi sidecar `<model>.gate.json` dengan `status: "eligible"` jika:

- manifest, provenance, dan split lengkap;
- tidak ada class leakage atau duplikasi lintas split yang terdeteksi;
- jumlah tiap kelas memenuhi minimum yang ditentukan saat audit;
- metrik minimum disetujui setelah melihat hasil nyata;
- test lokal dan test lintas-session tidak menunjukkan kegagalan fatal;
- model tidak mengubah output task lain.

Jika syarat belum terpenuhi, statusnya `rejected` atau `blocked`, bukan `eligible`.

## Integrasi aplikasi

Sebelum model tersedia, UI tetap menampilkan:

| Fitur | Status |
|---|---|
| Penyakit | Belum tersedia |
| Kematangan | Belum tersedia |
| Busuk | Belum tersedia |
| Grade kualitas | Belum tersedia |

Setelah satu model lolos gate, hanya fitur itu yang berubah menjadi tersedia. Fitur lain
tetap `Belum tersedia`. Pesan aplikasi harus membedakan `indikasi model` dari diagnosis
ahli dan tetap menunjukkan confidence serta status review.

## Non-goals dan anti-goals

- Tidak memakai detector jenis buah sebagai label penyakit, matang, busuk, atau grade.
- Tidak mengubah label `ripe` menjadi jumlah hari.
- Tidak mengunduh dataset besar atau mengirim foto pengguna ke provider tanpa persetujuan.
- Tidak memakai hasil AI generatif sebagai ground truth.
- Tidak mempromosikan model dari tanaman lain sebagai model untuk tiga tanaman target.
- Tidak menghentikan loop detector yang sedang berjalan.

## Persetujuan yang diperlukan

Sebelum implementasi, pengguna perlu menyetujui desain ini dan menyediakan salah satu:

1. folder dataset berlabel milik pengguna, atau
2. izin eksplisit untuk mengunduh dataset publik yang lisensinya sudah diverifikasi.

Tanpa salah satu sumber data tersebut, yang dapat dibuat secara jujur hanya scaffold,
audit, dan status `blocked_no_dataset`, bukan model yang siap dipakai.
