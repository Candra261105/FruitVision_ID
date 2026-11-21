from __future__ import annotations
import sys
from pathlib import Path
from ultralytics import YOLO
from research.detector_training import find_best_weight
data, weights, out, epochs, imgsz, batch, name = sys.argv[1:8]
model = YOLO(weights)
model.train(data=data, epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), workers=0,
            device='cpu', project=out, name=name, exist_ok=True, patience=4, plots=False,
            val=True, seed=0, cache='ram', verbose=True)
best = find_best_weight(Path(out), name, Path.cwd())
print('BEST', best, best.stat().st_size)
