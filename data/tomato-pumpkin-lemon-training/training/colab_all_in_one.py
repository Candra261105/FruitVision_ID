#!/usr/bin/env python3
"""
FruitVision ID - All-in-One Training, Evaluation & Auto-Download Pipeline
Menjalankan training otomatis di Google Colab (Tesla T4 GPU), menampilkan grafik &
persentase akurasi, lalu mengunduh model terbaik (best.pt) dan grafik ke laptop.
"""

import os
import glob
import time
from pathlib import Path
from ultralytics import YOLO

# 1. SETUP LOKASI & DEPENDENSI
workspace_dir = Path("/content/tomato-pumpkin-lemon-training")
if workspace_dir.exists():
    os.chdir(str(workspace_dir))
    data_yaml = str(workspace_dir / "dataset" / "data.yaml")
else:
    # Fallback lokal jika diuji di luar Colab
    workspace_dir = Path(__file__).resolve().parent.parent
    os.chdir(str(workspace_dir))
    data_yaml = str(workspace_dir / "dataset" / "data.yaml")

print("=" * 60)
print("🍎 FRUITVISION_ID - PIPELINE ALL-IN-ONE (TRAIN -> EVAL -> DOWNLOAD)")
print(f"Direktori Kerja: {os.getcwd()}")
print("=" * 60)

# Cek checkpoint terbaik sebelumnya jika ada, atau gunakan bobot bawaan
candidate_weights = glob.glob(str(workspace_dir / "runs/**/weights/best.pt"), recursive=True)
candidate_weights += glob.glob("/content/**/weights/best.pt", recursive=True)
candidate_weights = list(set(candidate_weights))

if candidate_weights:
    base_model = sorted(candidate_weights)[-1]
    print(f"[*] Melanjutkan pembelajaran dari model terbaik sebelumnya: {base_model}")
else:
    default_pt = workspace_dir / "yolo26n.pt"
    base_model = str(default_pt) if default_pt.exists() else "yolo11n.pt"
    print(f"[*] Memulai training dari model dasar: {base_model}")

# 2. EKSEKUSI TRAINING
print("\n" + "=" * 60)
print("🚀 MEMULAI TRAINING (BATCH 32, CACHE RAM, AUTO-STOP PATIENCE 40)")
print("=" * 60)

model = YOLO(base_model)
results = model.train(
    data=data_yaml,
    epochs=1000,
    imgsz=640,
    batch=32,            # Keseimbangan emas: presisi detail tinggi & kecepatan optimal
    cache="ram",         # Memuat dataset ke RAM agar nol hambatan disk
    workers=4,           # Optimasi multi-core CPU Colab
    device=0,            # Tesla T4 GPU
    patience=40,         # Berhenti otomatis saat akurasi sudah datar di puncak
    project=str(workspace_dir / "runs" / "adaptive"),
    name="adaptive-all-in-one",
    exist_ok=True,
    save_period=10,
    plots=True           # Menghasilkan grafik visual (results.png, confusion_matrix.png)
)

save_dir = Path(model.trainer.save_dir)
best_weight = save_dir / "weights" / "best.pt"
print(f"\n[✓] Training selesai! File bobot terbaik: {best_weight}")

# 3. EVALUASI AKURASI DETAIL
print("\n" + "=" * 60)
print("📊 HASIL EVALUASI AKURASI MODEL TERBAIK")
print("=" * 60)

eval_model = YOLO(str(best_weight))
metrics = eval_model.val(data=data_yaml, device=0)

overall_map50 = metrics.box.map50 * 100
overall_map = metrics.box.map * 100
mp = metrics.box.mp * 100
mr = metrics.box.mr * 100

print(f"🏆 AKURASI KESELURUHAN (mAP50): {overall_map50:.2f}%")
print(f"📈 mAP50-95 (Standar Ketat)  : {overall_map:.2f}%")
print(f"🎯 Rata-rata Presisi (P)     : {mp:.2f}%")
print(f"🔍 Rata-rata Recall (R)      : {mr:.2f}%")
print("-" * 60)
print("DETAIL AKURASI PER KELAS BUAH:")
for idx, name in eval_model.names.items():
    skor_class = metrics.box.class_result(idx)[2] * 100
    print(f" • {name.capitalize():<10} : {skor_class:.2f}%")
print("=" * 60)

# 4. TAMPILKAN GRAFIK DI LAYAR
print("\n" + "=" * 60)
print("📈 MENAMPILKAN GRAFIK HASIL TRAINING")
print("=" * 60)

try:
    from IPython.display import Image, display
    results_png = save_dir / "results.png"
    if results_png.exists():
        print("\n[Grafik 1: Kurva Penurunan Loss & Kenaikan Akurasi (mAP50)]")
        display(Image(filename=str(results_png), width=900))

    cm_png = save_dir / "confusion_matrix.png"
    if cm_png.exists():
        print("\n[Grafik 2: Confusion Matrix (Matriks Kebenaran Deteksi Tiap Buah)]")
        display(Image(filename=str(cm_png), width=700))
except Exception as err:
    print(f"[Info] Tampilan inline dilewati: {err}")

# 5. DOWNLOAD OTOMATIS KE LAPTOP
print("\n" + "=" * 60)
print("📥 MENGUNDUH MODEL & GRAFIK KE LAPTOP SECARA OTOMATIS...")
print("=" * 60)

try:
    from google.colab import files
    if best_weight.exists():
        print(f"[*] Mengunduh model bobot: {best_weight.name} ...")
        files.download(str(best_weight))
        time.sleep(2)

    results_png = save_dir / "results.png"
    if results_png.exists():
        print(f"[*] Mengunduh grafik pelatihan: {results_png.name} ...")
        files.download(str(results_png))

    print("\n✅ SELESAI SEMPURNA! Silakan cek folder Downloads di laptop Anda.")
except ImportError:
    print("[Info] Bukan lingkungan Google Colab. File tersimpan aman di disk lokal.")
