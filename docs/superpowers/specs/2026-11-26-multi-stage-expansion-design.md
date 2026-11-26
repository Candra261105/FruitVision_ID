# Desain Fusi Tiga Tahap: Ekspansi Multi-Sumber, AI Re-Cropping, Auto-Labeling & YOLO11m

## 1. Latar Belakang & Evaluasi Baseline (YOLO11s @ 104 Epoch)
Evaluasi benchmark independen (300 val images, 2.266 instances) membuktikan:
- **Pumpkin**: **94.94% mAP50** (Kuat dan stabil, hanya perlu dipertahankan).
- **Tomato**: **68.22% mAP50** (Akurasi tertahan akibat tumpang-tindih ekstrem dan variasi warna tomat hijau/merah).
- **Lemon**: **56.13% mAP50** (Paling lemah; dataset sebelumnya kekurangan variasi lemon kuning, lemon meja, dan variasi skala).
- **Colab Download Handler**: `AttributeError: 'NoneType' object has no attribute 'kernel'` terjadi ketika `google.colab.files.download()` dipanggil di lingkungan skrip terminal subprocess.

## 2. Sasaran (Goals) & Batasan
1. **Peningkatan Akurasi**: Menargetkan peningkatan rata-rata mAP50 mendekati 90–95%+ dengan melipatgandakan data latih lemon dan tomat secara signifikan.
2. **AI Re-cropping & Quality Refinement**: Menyeleksi dan mengekstrak crop buah berkualitas tinggi dengan padding adaptif dan tepi feathered halus dari bank data lokal (60k tomat, lemon LVIS/worldwide).
3. **Web Harvesting & Direct AI Auto-Labeling**: Mengambil citra pemandangan nyata (dapur, pasar, supermarket) dan melabeli koordinat bounding box secara otomatis via detektor AI dengan filter confidence $\ge 0.60$.
4. **Leakage-Guarded**: Split `val` (300) dan `test` (300) **wajib tetap beku 100%**.
5. **Colab Runner Hardening & YOLO11m**:
   - Memperbaiki bug download Colab dengan fallback multi-opsi (Auto-Zip, Google Drive copy, dan direct download try-except).
   - Meng-upgrade model ke `yolo11m.pt` (Medium: 20M params) yang muat optimal di Tesla T4 VRAM 16GB.

## 3. Tiga Tahap Sinergis
1. **Tahap 1**: Targeted Weak-Class Harvesting & AI Re-cropping (`research/harvest_fruit_crops.py` + `research/ai_refine_crops.py`).
2. **Tahap 2**: Web Harvester & AI Auto-Labeling (`research/harvest_web_context.py` + `research/ai_auto_label.py`).
3. **Tahap 3**: High-Grade Composite Fusion, Leakage Verification, Colab Fix & Package (`research/generate_fruit_composites.py`, `research/build_expanded_dataset.py`, `colab_train_expanded.py`, `package_expanded_colab.py`).
