# Multi-Stage Dataset Expansion & AI Auto-Labeling Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menghasilkan dataset terluas (real multi-source + AI re-cropped + auto-labeled web scenes + high-grade composites) dan memperbarui runner Colab YOLO11m bebas bug unduhan untuk mendongkrak akurasi deteksi tomat, labu, dan lemon mendekati 90–95%+ mAP50.

**Architecture:** Membangun pipeline 3 tahap: (1) AI Re-cropping & weak-class harvesting (lemon & tomato dari bank lokal), (2) Web context harvester & AI auto-labeling untuk pemandangan nyata, (3) Fusi komposit masif + pembekuan benchmark val/test + runner Colab YOLO11m yang tahan gangguan subshell Colab.

**Tech Stack:** Python 3.11+, Ultralytics YOLO11, Pillow, NumPy, PyYAML, Pytest.

**Spec:** `docs/superpowers/specs/2026-11-26-multi-stage-expansion-design.md`

## Global Constraints
- Target kelas wajib: `0: tomato`, `1: pumpkin`, `2: lemon`.
- Split benchmark `val` (300 citra) dan `test` (300 citra) wajib 100% citra nyata dan dibekukan tanpa kebocoran (*zero leakage*).
- Komputasi lokal ramah CPU Ryzen 5 3500U: tidak melatih YOLO lokal, komputasi berat dipaketkan ke Google Colab T4 GPU.
- Penanganan download Colab tidak boleh gagal ketika dijalankan via subshell subprocess/terminal.

## Review Focus
- Bounding box out of bounds ($<0.0$ atau $>1.0$) pada auto-labeling harus otomatis dipangkas (*clipped*) atau ditolak.
- Label duplikat pada koordinat identik harus dieliminasi via NMS/deduplication.
- Auto-labeling confidence threshold minimal 0.55 untuk mencegah false positive buah lain.
- Penanganan error unduhan Colab harus memiliki fallback ke pembuatan bundle zip lokal dan direktori Drive.
- Integritas backend FruitVision (`backend/tests/`) harus tetap 100% lulus tanpa regresi.

---

### Task 1: AI Crop Harvester & Refinement Engine (Tahap 1)

**Files:**
- Create: `research/ai_refine_crops.py`
- Modify: `research/harvest_fruit_crops.py`
- Test: `tests/test_ai_refine_crops.py`

**Interfaces:**
- Consumes: `data/advanced/source/tomato_ripeness_crops.zip`, `data/raw/lvis-fruits-vegetables-github/lvis.zip`, `data/processed/bangladesh/`
- Produces: `refine_and_feather_crop(img, mask) -> PIL.Image`, `extract_tomato_ripeness_crops(zip_path, out_dir, limit)`

- [ ] **Step 1: Write test for crop refinement and feathering**

```python
def test_refine_and_feather_crop():
    from PIL import Image
    from research.ai_refine_crops import refine_and_feather_crop
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    refined = refine_and_feather_crop(img, feather_radius=3)
    assert refined.mode == "RGBA"
    assert refined.size == (100, 100)
    # Check corner alpha is 0 or feathered
    alpha = refined.split()[-1]
    assert alpha.getpixel((0, 0)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_refine_crops.py -v`
Expected: FAIL with ModuleNotFoundError or ImportError

- [ ] **Step 3: Implement `research/ai_refine_crops.py` and enhance `research/harvest_fruit_crops.py`**

Implementasikan ekstraksi tomat dari `tomato_ripeness_crops.zip`, lemon dari dataset tersedia, serta fungsi penghalusan sudut (*feathered circular/oval alpha mask*) untuk mencegah artefak sudut hitam kotak.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ai_refine_crops.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/ai_refine_crops.py research/harvest_fruit_crops.py tests/test_ai_refine_crops.py
git commit -m "feat: add ai crop harvester and feathering refinement engine"
```

---

### Task 2: Web Context Harvester & AI Auto-Labeling Engine (Tahap 2)

**Files:**
- Create: `research/harvest_web_context.py`
- Create: `research/ai_auto_label.py`
- Test: `tests/test_ai_auto_label.py`

**Interfaces:**
- Consumes: Target URLs / CC public datasets / synthetic scene templates
- Produces: `filter_and_deduplicate_boxes(boxes, iou_thresh=0.5) -> list[tuple]`, `generate_yolo_label_lines(detections, img_w, img_h) -> str`

- [ ] **Step 1: Write test for auto-label deduplication and box filtering**

```python
def test_filter_and_deduplicate_boxes():
    from research.ai_auto_label import filter_and_deduplicate_boxes
    # class, cx, cy, w, h, conf
    boxes = [
        (0, 0.5, 0.5, 0.2, 0.2, 0.9),
        (0, 0.501, 0.501, 0.199, 0.199, 0.85), # duplicate
        (2, 0.1, 0.1, 0.05, 0.05, 0.4), # low conf
        (1, 0.8, 0.8, 0.3, 0.3, 0.95),
    ]
    filtered = filter_and_deduplicate_boxes(boxes, min_conf=0.6, iou_thresh=0.5)
    assert len(filtered) == 2
    assert filtered[0][0] == 0
    assert filtered[1][0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ai_auto_label.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `research/ai_auto_label.py` and `research/harvest_web_context.py`**

Implementasikan IoU matching, deduplikasi kotak bertumpuk (*redundant overlapping boxes*), validasi rentang koordinat 0.0–1.0, serta pengunduh/penyusun citra konteks dapur/rak nyata.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ai_auto_label.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/harvest_web_context.py research/ai_auto_label.py tests/test_ai_auto_label.py
git commit -m "feat: implement web context harvesting and ai auto-labeling engine"
```

---

### Task 3: Large-Scale Composite Fusion & Dataset Assembler (Tahap 3)

**Files:**
- Modify: `research/generate_fruit_composites.py`
- Modify: `research/build_expanded_dataset.py`
- Test: `tests/test_dataset_expansion.py`

**Interfaces:**
- Consumes: Bank crop halus (Tahap 1), Real & Auto-labeled scenes (Tahap 2), Benchmark splits
- Produces: Dataset `train`, `val` (300 frozen), `test` (300 frozen) di `data/processed/tomato-pumpkin-lemon-expanded/dataset`

- [ ] **Step 1: Write integration test for dataset expansion scaling**

Pastikan test memverifikasi integritas koordinat, zero synthetic leakage ke `val`/`test`, dan peningkatan kuantitas instance lemon/tomato.

- [ ] **Step 2: Run test to verify current state**

Run: `uv run pytest tests/test_dataset_expansion.py -v`
Expected: PASS (pada state sebelumnya)

- [ ] **Step 3: Update `generate_fruit_composites.py` & `build_expanded_dataset.py`**

Tingkatkan variasi komposit (target 2.000–3.000 citra komposit beresolusi 800x800), integrasikan citra scene konteks baru, dan bobotkan proporsi objek lemon & tomat.

- [ ] **Step 4: Generate expanded dataset & verify leakage**

Jalankan generator komposit dan assembler dataset, lalu pastikan verifikasi kebocoran (*leakage guard*) menyatakan lolos 100%.

- [ ] **Step 5: Commit**

```bash
git add research/generate_fruit_composites.py research/build_expanded_dataset.py tests/test_dataset_expansion.py
git commit -m "feat: scale composite generation and integrate multi-source dataset fusion"
```

---

### Task 4: Colab Training Runner Hardening & YOLO11m Upgrade

**Files:**
- Modify: `data/tomato-pumpkin-lemon-expanded/training/colab_train_expanded.py`
- Modify: `docs/COLAB_EXPANDED_GUIDE.md`
- Test: `tests/test_colab_runner.py`

**Interfaces:**
- Consumes: `dataset/data.yaml`
- Produces: Pelatihan YOLO11m @ 800px dengan auto-download bebas error dan opsi bundle zip artifact.

- [ ] **Step 1: Write test for Colab runner download fallback**

```python
def test_colab_download_fallback_logic():
    from research.colab_download_helper import safe_download_artifacts
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "best.pt"
        f.write_text("dummy weight")
        # Should not raise AttributeError even if IPython is None
        status = safe_download_artifacts([f], zip_name=Path(tmp) / "bundle.zip")
        assert status["zip_created"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_colab_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `research/colab_download_helper.py` and patch `colab_train_expanded.py`**

- Tangani `AttributeError` secara eksplisit saat `_IPython.get_ipython().kernel` bernilai `None`.
- Buat bundle zip ringkas `fruitvision_yolo11m_best_artifacts.zip` berisi `best.pt`, `results.png`, `confusion_matrix.png`, dan metrik.
- Perbarui konfigurasi default model ke `yolo11m.pt` (Medium), imgsz 800, batch 16, epochs 1000, patience 50.
- Perbarui panduan `docs/COLAB_EXPANDED_GUIDE.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_colab_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research/colab_download_helper.py data/tomato-pumpkin-lemon-expanded/training/colab_train_expanded.py docs/COLAB_EXPANDED_GUIDE.md tests/test_colab_runner.py
git commit -m "fix: harden colab auto-download handler and upgrade training to yolo11m"
```

---

### Task 5: Packaging & End-to-End Verification

**Files:**
- Modify: `research/package_expanded_colab.py`
- Output: `data/tomato-pumpkin-lemon-expanded.zip`

- [ ] **Step 1: Execute package builder**

Run: `uv run python research/package_expanded_colab.py`
Expected: Sukses membuat `data/tomato-pumpkin-lemon-expanded.zip`.

- [ ] **Step 2: Run all unit and regression tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 3: Run full backend regression tests**

Run: `uv run pytest backend/tests/ -v`
Expected: All 147 backend tests PASS.

- [ ] **Step 4: Commit**

```bash
git add research/package_expanded_colab.py docs/
git commit -m "chore: package expanded multi-stage dataset for colab yolo11m training"
```
