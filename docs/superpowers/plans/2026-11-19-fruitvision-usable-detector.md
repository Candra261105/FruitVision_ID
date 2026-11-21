# FruitVision Usable Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Membuat FruitVision lebih layak dipakai untuk mengenali `cucumber`, `tomato`, dan `green beans`, memperjelas gambar, serta memberi peringatan jujur untuk gambar crop/terpotong.

**Architecture:** Runtime melakukan enhancement dan inferensi detector utama, lalu mencoba jalur fallback multi-skala hanya saat hasil kosong. Dataset detector dibangun dari sumber berlabel yang diaudit, sementara pseudo-label AI hanya dipakai sebagai data training terpisah dan tidak masuk validation/test. Model custom hanya dipromosikan setelah evaluasi test lama dan holdout eksternal memenuhi gate.

**Tech Stack:** Python 3.11, FastAPI, Pillow, Ultralytics YOLO/YOLO-World, pytest, React/Vite.

**Spec:** `docs/superpowers/specs/2026-11-18-fruitvision-id-design.md` dan keputusan proyek FruitVision pada konteks sesi.

## Global Constraints

- Detector custom hanya dipromosikan jika mAP50 keseluruhan >= 0.55 dan AP50 setiap kelas >= 0.40.
- Dataset penyakit daun tidak boleh dipakai sebagai bukti penyakit buah.
- Augmentasi tidak masuk validation/test.
- Confidence rendah, objek crop, dan label tidak pasti harus berstatus `review`.
- Pseudo-label AI tidak boleh masuk test dan tidak boleh dianggap ground truth tanpa audit.
- Training CPU-only, resumable, mendukung `STOP_TRAINING`, dan tidak menjalankan training ganda.
- Tidak boleh mengklaim akurasi konsisten pada foto pengguna sebelum test lama, external holdout, dan uji aplikasi selesai.

## Review Focus

- Crop 160x160 yang memenuhi seluruh frame: runtime harus tetap mencoba penilaian kondisi tanpa menganggapnya bounding box detector yang valid.
- Tomat jelas tetapi detector kosong: fallback multi-skala harus dijalankan dan hasil kosong harus menjelaskan keterbatasan model.
- Objek menyentuh tepi: hasil boleh ditampilkan tetapi harus memiliki peringatan `objek menyentuh tepi` dan status review bila keyakinan rendah.
- Gambar gelap/noise/silau: enhancement harus memperbaiki salinan inferensi dan menampilkan langkahnya tanpa mengganti gambar asli.
- Pseudo-label salah atau duplikat: builder harus memisahkan data review/train dari validation/test dan mencatat provenance.

---

### Task 1: Runtime detection fallback dan crop handling

**Files:**
- Modify: `backend/app/inference/ultralytics_provider.py`
- Modify: `backend/app/services/prediction_service.py`
- Modify: `backend/app/services/image_enhancement.py`
- Test: `backend/tests/test_prediction_service.py`
- Test: `backend/tests/test_image_enhancement.py`

**Interfaces:**
- `UltralyticsProvider.predict_variants(image: Image.Image) -> list[list[RawDetection]]` menghasilkan prediksi dari gambar penuh, padded square, dan tile hanya bila diperlukan.
- `PredictionService` menggabungkan kandidat tanpa membuat prediksi palsu, mempertahankan bbox ternormalisasi dan warnings.
- Enhancement tetap mengembalikan `(Image.Image, EnhancementReport)`.

- [ ] Tulis test gagal untuk gambar crop kecil yang tetap menghasilkan jalur kondisi, gambar kosong yang tetap kosong, dan kandidat fallback yang dideduplikasi.
- [ ] Jalankan test dan pastikan gagal karena interface fallback belum ada.
- [ ] Implementasikan padded-square/multi-scale fallback yang hanya dijalankan saat detector utama kosong atau confidence rendah.
- [ ] Tambahkan deteksi heuristik non-klasifikasi untuk menandai crop/tepi berdasarkan ukuran dan posisi bbox, bukan untuk menciptakan kelas.
- [ ] Tambahkan enhancement terbatas untuk blur/noise/kontras tanpa merusak gambar asli yang dikembalikan ke UI.
- [ ] Jalankan test runtime dan seluruh backend test.

### Task 2: Dataset detector tiga kelas dan pseudo-label terkontrol

**Files:**
- Modify: `research/build_detector_experiment.py`
- Create: `research/pseudo_label_detector.py`
- Create: `research/audit_detector_dataset.py`
- Modify: `backend/tests/test_research_tools.py`
- Create: `research/README_detector.md`

**Interfaces:**
- `build_detector_experiment(...)` menghasilkan split train/val/test deterministik tanpa collision dan tanpa stale files.
- `pseudo_label_detector.py` menulis label hanya ke folder kandidat/review dengan confidence, model, dan source image.
- `audit_detector_dataset.py` menghasilkan laporan label, bbox invalid, hash duplikat, provenance, dan leakage antar-split.

- [ ] Tulis test gagal untuk pemisahan pseudo-label dari test, duplicate hash lintas split, dan bbox terpotong yang diberi metadata.
- [ ] Jalankan test dan pastikan gagal pada aturan baru.
- [ ] Perbaiki builder agar sumber detector lama, CC0 tomato, dan data kandidat dapat digabung tanpa salah mapping kelas.
- [ ] Tambahkan pseudo-labeling lokal memakai detector/AI yang tersedia, dengan threshold tinggi dan folder `review` terpisah.
- [ ] Audit semua gambar dan keluarkan manifest final yang memuat provenance dan lisensi.
- [ ] Jalankan test riset dan audit dataset sebelum training.

### Task 3: Training dan evaluasi detector custom

**Files:**
- Modify: `runs/loop/train_detector_once.py`
- Modify: `research/fruitvision_loop.py`
- Create: `runs/loop/evaluate_detector_final.py`
- Create: `runs/loop/promote_detector.py`
- Modify: `runs/loop/REPORT.md`

**Interfaces:**
- Training wrapper mencari output Ultralytics di lokasi aktual, dapat resume, dan mengembalikan exit code yang benar.
- Evaluator menulis metrik test lama dan external holdout dengan tiga kelas yang sama.
- Promoter hanya menyalin bobot ke `models/detector.pt` dan membuat `models/detector.gate.json` bila seluruh gate terpenuhi.

- [ ] Tulis test gagal untuk pencarian `best.pt` di `runs/detect`, penolakan mapping kelas yang tidak lengkap, dan gate AP50 per kelas.
- [ ] Jalankan test dan pastikan gagal pada wrapper/promoter yang belum lengkap.
- [ ] Perbaiki wrapper dan jalankan training CPU dengan checkpoint/resume serta guard training ganda.
- [ ] Evaluasi test lama, external holdout, dan subset foto uji manual tanpa memakai hasil pseudo-label sebagai test.
- [ ] Promosikan hanya bila mAP50 >= 0.55 dan AP50 setiap kelas >= 0.40, jika tidak biarkan fallback tetap aktif.
- [ ] Perbarui laporan dengan angka, hash bobot, sumber data, dan batasan domain.

### Task 4: Uji end-to-end dan packaging penggunaan

**Files:**
- Modify: `frontend/src/components/ResultPanel.tsx`
- Modify: `frontend/src/types.ts`
- Modify: `backend/tests/test_api.py`
- Modify: `README.md`
- Create: `Downloads/FruitVision_Test_Images/README.md` only if the fixture set changes.

**Interfaces:**
- API tetap mengembalikan `PredictionResponse` dengan original/enhanced image, warnings, review status, capabilities, dan conditions.
- UI menampilkan alasan tidak terdeteksi, status review crop/tepi, serta model yang aktif.

- [ ] Tulis test gagal untuk response kosong yang menjelaskan fallback/model, warning crop/tepi, dan enhanced image yang tetap tersedia.
- [ ] Jalankan test dan pastikan gagal sebelum perubahan UI/API.
- [ ] Implementasikan tampilan warning dan status model tanpa menyatakan gambar gagal sebagai kelas tertentu.
- [ ] Jalankan backend test, frontend test, frontend production build, dan upload smoke test dengan 14 gambar uji.
- [ ] Hanya nyatakan proyek layak pakai pada ruang lingkup yang dibuktikan oleh hasil tersebut.
