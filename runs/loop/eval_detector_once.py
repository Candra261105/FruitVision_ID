from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
from ultralytics import YOLO
weights, data, imgsz, out = sys.argv[1:5]
model = YOLO(weights)
names = model.names
metrics = model.val(data=data, split='test', imgsz=int(imgsz), device='cpu', workers=0, plots=False, verbose=False)
box = metrics.box
per_class = {}
ap50 = getattr(box, 'ap50', None)
index = list(getattr(box, 'ap_class_index', []))
if ap50 is not None and len(ap50) == len(index):
    for position, cls in enumerate(index):
        per_class[names[int(cls)]] = float(ap50[position])
else:
    for position, cls in enumerate(index):
        per_class[names[int(cls)]] = float(box.maps[position])
payload = {'map50': float(box.map50), 'map50_95': float(box.map), 'per_class': per_class,
           'precision': float(box.mp), 'recall': float(box.mr)}
Path(out).write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload, indent=2))
