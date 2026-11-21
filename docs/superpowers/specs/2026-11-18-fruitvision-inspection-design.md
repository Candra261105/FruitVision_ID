# Desain FruitVision ID: Inspeksi Tanaman dan Kualitas Produk

Tanggal: 18 November 2026

## Intent

FruitVision ID menerima foto dari kamera atau upload, memperjelas gambar secara otomatis, mendeteksi objek yang didukung, menilai kualitas foto dan objek, lalu menyajikan penjelasan yang dapat dipakai petani atau pabrik untuk pemeriksaan awal.

Hasil harus jujur tentang tingkat keyakinan. Sistem tidak boleh menyatakan penyakit, tingkat kematangan, atau kondisi busuk sebagai fakta jika model dan data berlabel untuk hal tersebut belum tersedia.

## Tujuan

- Memakai jalur pemrosesan yang sama untuk kamera dan upload.
- Menampilkan perbandingan gambar asli dan hasil enhancement.
- Memakai gambar enhanced untuk inference agar pencahayaan buruk tidak langsung menggagalkan deteksi.
- Menyediakan metrik kualitas foto: exposure, kontras, blur, noise, silau, dan ukuran objek.
- Mendeteksi tiga kelas detector saat ini: `cucumber`, `tomato`, dan `green beans`.
- Menolak confidence rendah dan hasil yang tidak masuk akal.
- Menyediakan rekomendasi tindakan yang bersumber dari hasil computer vision.
- Menambahkan AI provider secara opsional untuk merangkum hasil, bukan menggantikan detector.
- Menyimpan API key hanya di environment backend, tidak di frontend dan tidak di repository.

## Non-goal dan batasan

- Detector tiga kelas tidak otomatis dapat menentukan penyakit.
- Detector objek tidak otomatis dapat menentukan umur hari, ukuran fisik, tingkat kematangan, atau busuk.
- AI vision umum tidak boleh diperlakukan sebagai diagnosis agronomi atau keputusan penerimaan pabrik tanpa validasi lapangan.
- Data kitchen tetap test lintas-domain dan tidak masuk training.
- Model belum boleh dipromosikan ke aplikasi sebelum melewati gate evaluasi per kelas dan Kitchen.

## Pendekatan yang dipertimbangkan

### 1. Enhancement di browser saja

Cepat dan mengurangi beban backend, tetapi hasil bisa berbeda antar-browser, kamera, dan perangkat. Tidak dipilih sebagai jalur utama.

### 2. Enhancement di backend tanpa preview

Konsisten dan mudah diuji, tetapi pengguna tidak bisa melihat apakah gambar berubah terlalu agresif. Tidak cukup transparan untuk inspeksi lapangan.

### 3. Dual path di backend, dengan preview asli dan enhanced

Foto divalidasi dan diproses di backend, model memakai hasil enhanced, sedangkan web menampilkan kedua versi beserta laporan perubahan. Ini dipilih karena konsisten, dapat diaudit, dan membantu pengguna memperbaiki pengambilan foto berikutnya.

## Alur sistem

```text
Kamera / Upload
    |
    v
Validasi ukuran, format, dimensi, dan pixel
    |
    v
Analisis kondisi gambar
    |
    v
Enhancement konservatif jika diperlukan
    |
    +--> preview asli dan enhanced
    |
    v
Detector YOLO pada gambar enhanced
    |
    v
Quality analyzer per gambar dan per objek
    |
    +--> model maturity/rot/disease jika sudah tersedia dan lolos gate
    |
    v
AI explainer opsional
    |
    v
Respons web: hasil, confidence, peringatan, dan rekomendasi
```

## Enhancement

Enhancement memakai pipeline yang sudah diuji di proyek AutoEnhancer UMKM, dengan urutan dan parameter yang ditulis pada laporan hasil:

1. Validasi RGB dan ukuran.
2. Analisis mean brightness, percentile kontras, noise, dan proporsi piksel silau.
3. Koreksi gamma/log/kontras hanya bila kondisi memenuhi ambang.
4. Denoise hanya bila noise terdeteksi.
5. Resize aman untuk inference.
6. Simpan laporan metode yang dipakai dan nilai sebelum/sesudah.

Enhancement tidak boleh selalu menaikkan ketajaman, karena sharpening berlebihan dapat membuat tepi palsu yang mengganggu detector. Jika gambar sudah layak, hasil enhanced dapat sama atau hampir sama dengan asli dan statusnya `unchanged`.

## Hasil detector

Setiap detection memuat:

- kelas internal dan nama tampilan
- confidence
- bounding box ternormalisasi
- ukuran relatif objek
- status confidence: `accepted`, `review`, atau `rejected`
- peringatan jika objek terlalu kecil, terpotong, buram, atau tertutup

Threshold awal tetap konservatif. Nilainya dapat dikalibrasi setelah evaluasi model final, bukan diturunkan hanya agar jumlah deteksi terlihat tinggi.

## Analisis kualitas

### Level gambar

- brightness: terlalu gelap atau terlalu terang
- contrast: cukup atau rendah
- blur: perkiraan ketajaman menggunakan ukuran yang terdokumentasi
- noise: indikasi noise sensor atau kompresi
- glare: area putih/silau yang dapat menutupi objek
- resolution: cukup untuk pemeriksaan atau perlu foto ulang

### Level objek

- area bounding box terhadap luas gambar
- objek menyentuh tepi gambar
- perkiraan occlusion dari bentuk box dan confidence
- crop terlalu kecil untuk analisis lanjutan

Laporan harus membedakan `quality_warning` dari `diagnosis`. Peringatan kualitas boleh dibuat dari aturan CV. Diagnosis penyakit tidak boleh dibuat dari aturan kualitas saja.

## Maturity, busuk, penyakit, dan kualitas produk

Fitur ini dibagi menjadi model terpisah:

| Fitur | Syarat sebelum diaktifkan |
|---|---|
| Kematangan | Dataset per kelas dengan label tahap yang konsisten dan provenance jelas |
| Busuk | Dataset busuk/tidak busuk dengan variasi pencahayaan dan background |
| Penyakit | Dataset penyakit per tanaman, label ahli atau sumber tepercaya, dan evaluasi per penyakit |
| Grade kualitas | Definisi grade dari pengguna/pabrik, label operator, dan ambang yang disetujui |
| Ukuran fisik | Kalibrasi skala, marker atau kamera terkontrol, bukan perkiraan dari satu foto bebas |

Sebelum syarat terpenuhi, web menampilkan `belum tersedia` atau `perlu pemeriksaan`, bukan nilai rekaan. Jika model tersedia tetapi confidence rendah, hasilnya `review_required`.

## Integrasi AI provider

AI provider bersifat opsional dan tidak memblokir detector utama.

Rekomendasi awal: adapter provider-neutral dengan OpenRouter sebagai provider pertama karena dapat memilih model vision melalui satu endpoint. Groq dapat ditambahkan sebagai provider kedua bila model vision yang digunakan tersedia dan hasilnya lolos evaluasi. Provider dipilih melalui konfigurasi environment, bukan hard-code.

Contoh konfigurasi lokal, tanpa nilai rahasia:

```text
FRUITVISION_AI_ENABLED=false
FRUITVISION_AI_PROVIDER=openrouter
OPENROUTER_API_KEY=
OPENROUTER_MODEL=
GROQ_API_KEY=
GROQ_MODEL=
```

Aturan keamanan:

- API key hanya dibaca backend.
- Frontend tidak pernah menerima API key.
- Key tidak ditulis ke log, response, atau file konfigurasi yang di-commit.
- AI dipanggil hanya jika user mengaktifkan fitur dan hasil CV tersedia.
- Timeout, retry terbatas, batas ukuran gambar, dan fallback tanpa AI wajib ada.
- Prompt mengandung data CV terstruktur dan meminta AI mengakui data yang tidak tersedia.
- Respons AI divalidasi ke schema. Teks bebas tidak boleh langsung menjadi status penyakit atau grade pabrik.

Output AI minimal:

- ringkasan temuan
- hal yang perlu diperiksa ulang
- rekomendasi foto atau tindakan berikutnya
- alasan singkat berdasarkan confidence dan quality warnings
- disclaimer bahwa hasil bukan diagnosis ahli jika menyangkut penyakit/keamanan pangan

## Tampilan web

Halaman hasil menampilkan:

1. Tab atau slider `Asli` dan `Enhanced`.
2. Canvas enhanced dengan bounding box.
3. Daftar objek, confidence, dan status review.
4. Panel kualitas foto dengan alasan yang mudah dipahami.
5. Panel kematangan/busuk/penyakit yang membedakan `tersedia`, `belum tersedia`, dan `perlu review`.
6. Panel rekomendasi untuk foto ulang, sortir, pemeriksaan manual, atau konsultasi ahli.
7. Label jelas jika AI provider tidak aktif atau gagal dipanggil.

## Tahapan implementasi

### Fase 1: Fondasi inference

- Selaraskan nama kelas backend dengan model tiga kelas.
- Pastikan model hanya dipakai setelah gate seleksi.
- Tambahkan schema enhancement report, quality report, dan model capability.

### Fase 2: Enhancement backend

- Implementasikan enhancement konservatif pada upload dan snapshot.
- Kembalikan preview enhanced atau referensi aman untuk frontend.
- Detector memakai gambar enhanced.
- Tambahkan test gambar gelap, terang, noise, dan gambar normal.

### Fase 3: Quality analyzer

- Tambahkan metrik kualitas gambar.
- Tambahkan peringatan per objek.
- Tampilkan rekomendasi foto ulang.

### Fase 4: Maturity/rot/disease/grade

- Audit kebutuhan label dan provenance.
- Jangan mengaktifkan model yang belum lolos evaluasi.
- Tambahkan model capability agar web tidak menampilkan fitur yang belum tersedia.

### Fase 5: AI explainer

- Tambahkan adapter OpenRouter.
- Tambahkan adapter Groq setelah model dan endpoint diverifikasi.
- Tambahkan validasi schema, timeout, redaction, dan fallback.
- Uji dengan AI mati, key salah, timeout, dan respons tidak sesuai schema.

### Fase 6: Evaluasi lapangan

- Kumpulkan foto pengguna dengan persetujuan.
- Pisahkan data train, validation, dan test berdasarkan gambar asli, bukan crop yang sama.
- Uji kondisi pasar, kebun, gudang, dan pabrik.
- Kalibrasi threshold dan buat laporan confusion matrix per kelas.

## Gate promosi

Tidak ada model atau fitur yang dipasang otomatis ke web jika:

- recall kelas di bawah ambang yang disepakati
- Kitchen atau test lintas-domain belum diuji
- label kelas tidak lengkap
- confidence belum dikalibrasi
- fitur penyakit/kematangan/grade belum memiliki data yang sesuai

Model smoke dan model yang gagal gate tetap hanya menjadi artefak penelitian.
