from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from app.inference.base import RawDetection


class UltralyticsProvider:
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float,
        class_prompts: tuple[str, ...] | None = None,
        imgsz: int = 640,
    ) -> None:
        if class_prompts:
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(str(model_path))
            self.model.set_classes(list(class_prompts))
            self._class_names = tuple(class_prompts)
            self.is_open_vocabulary = True
        else:
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
            self._class_names = None
            self.is_open_vocabulary = False
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz

    @property
    def class_names(self) -> tuple[str, ...]:
        if self._class_names is not None:
            return self._class_names
        names = self.model.names
        values = names.values() if isinstance(names, dict) else names
        return tuple(str(value) for value in values)

    def __call__(self, image: Image.Image) -> list[RawDetection]:
        detections = self._predict(image, self.confidence_threshold)
        if detections:
            return detections

        # YOLO-World sering gagal pada crop kecil tanpa konteks. Retry hanya
        # saat jalur utama kosong, supaya runtime tidak menurunkan threshold
        # pada setiap foto dan membanjiri hasil dengan false positive.
        width, height = image.size
        # Ukuran besar tetap dicoba pada frame asli dengan threshold review.
        # Padding hanya dibutuhkan untuk crop kecil yang kehilangan konteks.
        if min(width, height) > 320 and max(width, height) >= 640:
            detections = self._predict(image, min(self.confidence_threshold, 0.20))
            if detections:
                return detections
            return self._predict_tiles(image)
        padding = max(64, max(width, height))
        contextual = ImageOps.expand(image.convert("RGB"), border=padding, fill=(114, 114, 114))
        contextual_detections = self._predict(contextual, min(self.confidence_threshold, 0.20))
        normalized: list[RawDetection] = []
        for detection in contextual_detections:
            x1, y1, x2, y2 = detection.box
            pw, ph = contextual.size
            left = (x1 * pw - padding) / width
            top = (y1 * ph - padding) / height
            right = (x2 * pw - padding) / width
            bottom = (y2 * ph - padding) / height
            normalized.append(
                RawDetection(
                    label=detection.label,
                    confidence=detection.confidence,
                    box=(max(0.0, left), max(0.0, top), min(1.0, right), min(1.0, bottom)),
                    stage=detection.stage,
                )
            )
        return normalized

    def _predict_tiles(self, image: Image.Image) -> list[RawDetection]:
        width, height = image.size
        tile_width = min(width, max(320, int(height * 1.25)))
        if tile_width >= width:
            return []
        starts = sorted({0, max(0, (width - tile_width) // 2), width - tile_width})
        detections: list[RawDetection] = []
        for left in starts:
            tile = image.crop((left, 0, left + tile_width, height))
            for detection in self._predict(tile, min(self.confidence_threshold, 0.20)):
                x1, y1, x2, y2 = detection.box
                detections.append(
                    RawDetection(
                        label=detection.label,
                        confidence=detection.confidence,
                        box=(
                            max(0.0, (left + x1 * tile_width) / width),
                            y1,
                            min(1.0, (left + x2 * tile_width) / width),
                            y2,
                        ),
                        stage=detection.stage,
                    )
                )
        return _deduplicate(detections)

    def _predict(self, image: Image.Image, confidence: float) -> list[RawDetection]:
        results = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=self.imgsz,
            verbose=False,
            device="cpu",
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        names = result.names
        detections: list[RawDetection] = []
        width, height = image.size
        for box, confidence, class_id in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            x1, y1, x2, y2 = box
            label = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
            detections.append(
                RawDetection(
                    label=str(label),
                    confidence=float(confidence),
                    box=(x1 / width, y1 / height, x2 / width, y2 / height),
                )
            )
        return detections


def _deduplicate(detections: list[RawDetection]) -> list[RawDetection]:
    selected: list[RawDetection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for current in selected:
            if candidate.label != current.label:
                continue
            ax1, ay1, ax2, ay2 = candidate.box
            bx1, by1, bx2, by2 = current.box
            intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
            if union and intersection / union >= 0.5:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return selected
