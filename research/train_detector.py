from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Latih detector YOLO setelah dataset dan data.yaml diaudit.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Path bobot lokal, misalnya D:\\model\\yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/fruitvision"))
    parser.add_argument("--allow-download", action="store_true", help="Izinkan unduhan bobot jika --model bukan file lokal.")
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"data.yaml tidak ditemukan: {args.data}")
    model_path = Path(args.model)
    if not args.allow_download and not model_path.is_file():
        raise SystemExit(
            f"Model lokal tidak ditemukan: {model_path}. "
            "Gunakan path file lokal atau tambahkan --allow-download secara eksplisit."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics belum terpasang. Pasang dengan: py -m pip install ultralytics") from exc
    model = YOLO(str(model_path) if model_path.is_file() else args.model)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, device=args.device, project=str(args.project))
    print("Training selesai. Salin bobot hasil evaluasi ke models/fruit_detector.pt setelah lolos audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
