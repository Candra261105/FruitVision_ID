# FruitVision ID - 35-Class Pretrained Production Model Integration Design

## Intent and Context
FruitVision ID is being upgraded to a production-ready fruit and vegetable detection system. Rather than requiring users to execute heavy deep learning training locally or download multi-gigabyte bundles from Google Drive, the system integrates a verified, lightweight (~38.7 MB) Ultralytics YOLO11m model pre-trained on 35 classes of fruits and vegetables (`Piyu12/fruit-veg-yolo11m-detector` from Hugging Face, mAP50: 92.5%).

## Goals
1. Make `models/fruit_veg_yolo11m.pt` the default detection model for the web application.
2. Enable full detection and reporting across all 35 classes (Apple, Banana, Orange, Tomato, Mango, Strawberry, etc.) with Indonesian localized names.
3. Remove stale Google Drive links from project documentation (`README.md` and `data/README.md`) and document the completed, ready-to-use version.
4. Provide auto-download fallback if `fruit_veg_yolo11m.pt` is missing.
5. Pass all unit and integration tests.
6. Push all commits to GitHub (`origin/main`).

## Non-Goals
- Altering the Colab training package (`data/tomato-pumpkin-lemon-expanded.zip`), which remains intact as an optional pathway for custom 3-class training.
- Training new models locally on CPU.

## Constraints
- Local hardware: ThinkPad E595 Ryzen 3500U Vega 8 (no CUDA). Inference must remain lightweight (<1s per image).
- Maintain existing API contract and compatibility with frontend React app.
