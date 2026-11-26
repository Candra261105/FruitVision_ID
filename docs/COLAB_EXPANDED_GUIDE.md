# Panduan Menjalankan Training Expanded (YOLO11x Maksimal @ 800px) di Google Colab T4

Panduan ini berisi langkah-langkah untuk melatih model deteksi buah **FruitVision ID (Tomato, Pumpkin, Lemon)** menggunakan paket dataset yang telah diperkaya (`tomato-pumpkin-lemon-expanded.zip`) dengan arsitektur maksimal untuk mencapai akurasi tertinggi di GPU Tesla T4 Google Colab.

---

## 1. Keunggulan Arsitektur & Dataset Baru
- **Arsitektur Maksimal (YOLO11x)**:
  * Menggunakan **`yolo11x.pt`** (Extra Large, 56,9M parameter, backbone C3k2 terdalam + C2PSA attention) secara default untuk mengejar akurasi puncak (mAP tertinggi).
  * Opsi fleksibel: dapat juga memilih **`yolo11m.pt`** (Medium, 20,1M parameter) jika ingin keseimbangan tinggi antara akurasi dan kecepatan CPU lokal.
- **Optimasi Khusus Tesla T4 (16GB VRAM)**:
  * Resolusi **`imgsz=800`** untuk menangkap lemon kecil dan tomat berkerumun.
  * Auto batch allocation: **`batch=16`** untuk YOLO11x (utilisasi VRAM ~85-90%, aman tanpa OOM) dan **`batch=32`** untuk YOLO11s/YOLO11m.
  * Mengaktifkan scheduler **Cosine LR (`cos_lr=True`)** dan **`close_mosaic=15`** agar koordinat bounding box di epoch akhir sangat presisi.
- **Total Objek Meningkat Drastis**:
  * **Lemon**: **6.737 objek** (sebelumnya sangat minim, kini menjadi prioritas utama).
  * **Tomato**: **9.947 objek** (mencakup variasi mentah, setengah matang, matang, dan meja dapur).
  * **Pumpkin**: **3.650 objek** (mempertahankan akurasi tinggi 95%+).
- **Latar Belakang Prosedural Cerdas**: Meja kayu, marmer dapur, kain lap, dan variasi pencahayaan alami (bukan kanvas kosong).
- **Benchmark Terkunci (100% Jujur)**: Data pengujian `val` (300 gambar) dan `test` (300 gambar) tetap menggunakan data asli yang sama persis tanpa bocor satu objek pun.

---

## 2. Langkah-Langkah di Google Colab

### Langkah 1: Buka Google Colab & Pilih GPU T4
1. Buka [Google Colab](https://colab.research.google.com/) dan buat Notebook Baru.
2. Di menu atas, klik **Runtime** > **Ubah jenis runtime** (*Change runtime type*).
3. Pilih **GPU T4** (Gratis).
4. Klik **Simpan**.

---

### Langkah 2: Unggah Dataset ke Colab
1. Di panel kiri Google Colab, klik ikon **Folder** (File).
2. Seret (*drag-and-drop*) file dari laptop Anda:
   `C:\Users\dragon\Downloads\FruitVision_ID\data\tomato-pumpkin-lemon-expanded.zip`
   ke panel folder Colab (`/content/`).
3. Tunggu hingga proses unggah selesai (lingkaran proses di bawah nama file selesai berputar).

---

### Langkah 3: Jalankan Pelatihan Otomatis (1 Sel All-in-One)
Buat satu sel kode di Colab, tempelkan seluruh kode di bawah ini, lalu jalankan (**Shift + Enter**):

```python
# 1. Ekstrak package
!unzip -q /content/tomato-pumpkin-lemon-expanded.zip -d /content/workspace
%cd /content/workspace

# 2. Install dependensi
!pip install -q ultralytics matplotlib

# 3. Eksekusi training YOLO11x Maksimal (800px, Auto-batch, Auto-eval, Auto-download)
!python training/colab_train_expanded.py
```

*Catatan: Jika ingin menggunakan varian Medium (20M parameter, lebih cepat di CPU laptop), cukup jalankan:*
```python
!python training/colab_train_expanded.py --model yolo11m.pt
```

---

### Langkah 4: Hasil Otomatis
Setelah model mencapai performa puncak (dihentikan otomatis oleh early stopping `patience=40`):
1. Bobot model terbaik (`best.pt`) akan **otomatis diunduh** oleh browser ke folder `Downloads` laptop Anda.
2. Grafik evaluasi (`results.png` dan `confusion_matrix.png`) ditampilkan langsung di output Colab.
3. Pindahkan file `best.pt` yang terunduh ke:
   `C:\Users\dragon\Downloads\FruitVision_ID\models\best.pt`
workspace_dir = Path("/content/tomato-pumpkin-lemon-expanded")
if not workspace_dir.exists():
    print("[*] Mengekstrak dataset expanded...")
    !unzip -q /content/tomato-pumpkin-lemon-expanded.zip -d /content/tomato-pumpkin-lemon-expanded

os.chdir(str(workspace_dir))
data_yaml = str(workspace_dir / "dataset" / "data.yaml")

from ultralytics import YOLO

# 3. Jalankan Script Pelatihan Terpadu
# (Atau langsung jalankan skrip bawaan package: !python training/colab_train_expanded.py)
print("\n" + "=" * 65)
print("🚀 MEMULAI TRAINING (YOLO11s, BATCH 32, IMGSZ 800, AUTO-STOP PATIENCE 40)")
print("=" * 65)

model = YOLO("yolo11s.pt")
results = model.train(
    data=data_yaml,
    epochs=1000,
    imgsz=800,           # Resolusi tinggi 800px untuk menangkap lemon & tomat kecil
    batch=32,            # Keseimbangan emas: presisi detail & stabilitas gradien VRAM
    cache="ram",         # Memuat 100% dataset ke RAM Colab (nol bottleneck disk)
    workers=4,           # Maksimalkan seluruh thread CPU Colab
    device=0,            # Tesla T4 GPU
    patience=40,         # Berhenti otomatis saat kurva akurasi sudah mencapai puncak
    project=str(workspace_dir / "runs" / "adaptive"),
    name="expanded-yolo11s-800px",
    exist_ok=True,
    save_period=10,
    plots=True
)

save_dir = Path(model.trainer.save_dir)
best_weight = save_dir / "weights" / "best.pt"

# 4. Tampilkan Skor Akurasi
print("\n" + "=" * 65)
print("📊 HASIL EVALUASI AKURASI MODEL TERBAIK")
print("=" * 65)

eval_model = YOLO(str(best_weight))
metrics = eval_model.val(data=data_yaml, device=0)

overall_map50 = metrics.box.map50 * 100
overall_map = metrics.box.map * 100
mp = metrics.box.mp * 100
mr = metrics.box.mr * 100

print(f"🏆 AKURASI KESELURUHAN (mAP50): {overall_map50:.2f}%")
print(f"📈 Standar Ketat (mAP50-95)  : {overall_map:.2f}%")
print(f"🎯 Presisi Rata-rata         : {mp:.2f}%")
print(f"🔍 Recall Rata-rata          : {mr:.2f}%")
print("-" * 65)
print("DETAIL AKURASI PER BUAH:")
for idx, name in eval_model.names.items():
    skor_class = metrics.box.class_result(idx)[2] * 100
    print(f" * {name.capitalize():<10} : {skor_class:.2f}%")
print("=" * 65)

# 5. Tampilkan Grafik Inline
try:
    from IPython.display import Image, display
    results_png = save_dir / "results.png"
    if results_png.exists():
        display(Image(filename=str(results_png), width=900))
except Exception as err:
    print(f"[Info] Tampilan inline dilewati: {err}")

# 6. Unduh Otomatis Model & Grafik ke Laptop
try:
    from google.colab import files
    if best_weight.exists():
        print(f"[*] Mengunduh model: {best_weight.name} ...")
        files.download(str(best_weight))
        time.sleep(2)
    results_png = save_dir / "results.png"
    if results_png.exists():
        print(f"[*] Mengunduh grafik pelatihan: {results_png.name} ...")
        files.download(str(results_png))
    print("\n[OK] SELESAI! File model dan grafik berhasil terunduh ke laptop Anda.")
except Exception as err:
    print(f"[Info] Download: {err}")
```

---

## 3. Memasang Model Baru ke Aplikasi FruitVision Lokal
Setelah file `best.pt` terunduh ke folder `Downloads` di laptop:
1. Pindahkan dan ganti nama file tersebut menjadi:
   `FruitVision_ID\.worktrees\tomato-pumpkin-lemon\models\tomato-pumpkin-lemon.pt`
2. Backend aplikasi FruitVision yang sedang berjalan akan otomatis memuat model baru berakurasi tinggi ini.
