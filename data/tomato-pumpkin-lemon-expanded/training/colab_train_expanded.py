#!/usr/bin/env python3
"""FruitVision ID - Expanded Dataset YOLO11s 800px Training & Auto-Download Pipeline.

Automated Colab execution script:
1. Auto-extracts tomato-pumpkin-lemon-expanded.zip if needed.
2. Trains YOLO11s at 800px resolution with batch=32, cache='ram', patience=40 on T4 GPU.
3. Evaluates mAP50 and per-fruit detection metrics on frozen benchmark.
4. Plots training curves and confusion matrix inline.
5. Triggers automatic download of best.pt and results.png to local Downloads.
"""

import glob
import os
import time
from pathlib import Path

from ultralytics import YOLO

# 1. SETUP WORKSPACE & DATASET
workspace_dir = Path("/content/tomato-pumpkin-lemon-expanded")
if workspace_dir.exists():
    os.chdir(str(workspace_dir))
    data_yaml = str(workspace_dir / "dataset" / "data.yaml")
else:
    # Check if zip exists in /content/
    zip_path = Path("/content/tomato-pumpkin-lemon-expanded.zip")
    if zip_path.exists():
        print("[*] Mengekstrak tomato-pumpkin-lemon-expanded.zip...")
        os.system(f"unzip -q {zip_path} -d /content/tomato-pumpkin-lemon-expanded")
        os.chdir(str(workspace_dir))
        data_yaml = str(workspace_dir / "dataset" / "data.yaml")
    else:
        # Fallback local testing path
        workspace_dir = Path(__file__).resolve().parent.parent
        os.chdir(str(workspace_dir))
        data_yaml = str(workspace_dir / "dataset" / "data.yaml")

print("=" * 65)
print("🍎 FRUITVISION ID - EXPANDED TRAINING PIPELINE (YOLO11s @ 800px)")
print(f"Direktori Kerja : {os.getcwd()}")
print(f"Data YAML       : {data_yaml}")
print("=" * 65)

# 2. SELECT MODEL (YOLO11s or RESUME CHECKPOINT)
candidate_weights = glob.glob(str(workspace_dir / "runs/**/weights/best.pt"), recursive=True)
candidate_weights += glob.glob("/content/**/weights/best.pt", recursive=True)
candidate_weights = list(set(candidate_weights))

if candidate_weights:
    base_model = max(candidate_weights, key=os.path.getmtime)
    print(f"[*] Melanjutkan pembelajaran dari checkpoint terbaik: {base_model}")
else:
    # Default to YOLO11s (Small, 9.4M params) for superior detail resolution
    base_model = "yolo11s.pt"
    print(f"[*] Memulai pelatihan baru dari arsitektur: {base_model}")

# 3. TRAINING ACCELERATION (TESLA T4 GPU)
print("\n" + "=" * 65)
print("🚀 MEMULAI TRAINING EXPANDED (BATCH 32, IMGSZ 800, PATIENCE 40)")
print("=" * 65)

model = YOLO(base_model)
results = model.train(
    data=data_yaml,
    epochs=1000,
    imgsz=800,           # Resolusi tinggi 800px untuk menangkap lemon & tomat kecil
    batch=32,            # Keseimbangan emas: presisi detail & stabilitas gradien VRAM
    cache="ram",         # Memuat 100% dataset ke RAM Colab (nol bottleneck disk)
    workers=4,           # Maksimalkan seluruh thread CPU Colab
    device=0,            # Tesla T4 GPU
    patience=40,         # Otomatis berhenti saat kurva akurasi sudah maksimal di puncak
    project=str(workspace_dir / "runs" / "adaptive"),
    name="expanded-yolo11s-800px",
    exist_ok=True,
    save_period=10,
    plots=True           # Menghasilkan grafik visual (results.png, confusion_matrix.png)
)

save_dir = Path(model.trainer.save_dir)
best_weight = save_dir / "weights" / "best.pt"
print(f"\n[OK] Pelatihan selesai! Bobot terbaik tersimpan di: {best_weight}")

# 4. EVALUASI AKURASI DETAIL PADA BENCHMARK
print("\n" + "=" * 65)
print("📊 EVALUASI AKURASI MODEL TERBAIK PADA BENCHMARK TERUJI")
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

# 5. TAMPILKAN GRAFIK DI LAYAR OUTPUT
print("\n" + "=" * 65)
print("📈 MENAMPILKAN GRAFIK EVALUASI & KURVA PEMBELAJARAN")
print("=" * 65)

try:
    from IPython.display import Image as IPImage, display
    results_png = save_dir / "results.png"
    if results_png.exists():
        print("\n[Grafik 1: Kurva Penurunan Loss & Kenaikan Akurasi]")
        display(IPImage(filename=str(results_png), width=900))

    cm_png = save_dir / "confusion_matrix.png"
    if cm_png.exists():
        print("\n[Grafik 2: Confusion Matrix (Ketepatan Prediksi per Buah)]")
        display(IPImage(filename=str(cm_png), width=700))
except Exception as err:
    print(f"[Info] Tampilan grafik inline dilewati: {err}")

# 6. AUTO-DOWNLOAD KE LAPTOP
print("\n" + "=" * 65)
print("📥 MENGUNDUH MODEL & GRAFIK KE LAPTOP SECARA OTOMATIS...")
print("=" * 65)

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

    print("\n[OK] SELESAI SEMPURNA! Silakan cek folder Downloads di laptop Anda.")
except ImportError:
    print("[Info] Bukan lingkungan Google Colab. File tersimpan aman di disk lokal.")
