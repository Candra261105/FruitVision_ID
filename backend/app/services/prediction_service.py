from __future__ import annotations

from collections.abc import Callable
from base64 import b64encode
from io import BytesIO
from math import isfinite

from PIL import Image

from app.inference.base import RawDetection
from app.schemas.prediction import (
    AIExplanation,
    Box,
    CapabilityStatus,
    ConditionAssessment,
    Detection,
    PredictionResponse,
    QualityReport,
)
from app.services.advanced_model_registry import AdvancedModelRuntime
from app.services.image_enhancement import EnhancementReport, enhance_image

FRUIT_NAMES = {
    "cucumber": "Mentimun",
    "tomato": "Tomat",
    "green_beans": "Buncis",
}

STAGE_NAMES = {
    "unripe": "mentah",
    "raw": "mentah",
    "half_ripe": "setengah matang",
    "ripe": "siap",
    "ready": "siap",
    "overripe": "terlalu matang",
    "rotten": "busuk",
}

CONDITION_DISPLAY = {
    "maturity": {"unripe": "mentah", "half_ripe": "setengah matang", "ripe": "matang"},
    "rot": {"not_rotten": "tidak terdeteksi busuk", "rotten": "busuk"},
    "disease": {
        "healthy": "tidak terdeteksi penyakit",
        "belly_rot": "busuk pangkal (belly rot)",
        "pythium_fruit_rot": "busuk pythium",
        "unknown": "jenis penyakit belum dikenali",
    },
    "grade": {"A": "grade A", "B": "grade B", "C": "grade C", "reject": "ditolak"},
}
CONDITION_MIN_CONFIDENCE = {"maturity": 0.55, "rot": 0.62, "disease": 0.60, "grade": 0.55}
CONDITION_MIN_MARGIN = {"maturity": 0.10, "rot": 0.15, "disease": 0.15, "grade": 0.12}
CONDITION_PADDING = 0.06
DEMO_REVIEW_THRESHOLD = 0.20


def split_label(raw_label: str, raw_stage: str | None) -> tuple[str, str | None]:
    label = raw_label.lower().strip().replace("-", "_").replace(" ", "_")
    stage = raw_stage.lower().strip() or None if raw_stage else None
    for fruit in FRUIT_NAMES:
        for separator in ("_", "/", ":"):
            prefix = f"{fruit}{separator}"
            suffix = f"{separator}{fruit}"
            if label.startswith(prefix):
                return fruit, stage or label[len(prefix) :]
            if label.endswith(suffix):
                return fruit, stage or label[: -len(suffix)]
        if label == fruit:
            return fruit, stage
    return label, stage


def _image_data_url(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + b64encode(buffer.getvalue()).decode("ascii")


def _report_to_schema(report: EnhancementReport) -> QualityReport:
    after = report.after
    return QualityReport(
        width=after.width,
        height=after.height,
        brightness=after.brightness,
        contrast=after.contrast,
        noise=after.noise,
        glare_ratio=after.glare_ratio,
        blur_score=after.blur_score,
        warnings=after.warnings,
    )


def _crop_detection(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image | None:
    width, height = image.size
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * CONDITION_PADDING
    pad_y = (y2 - y1) * CONDITION_PADDING
    left = max(0, int((x1 - pad_x) * width))
    top = max(0, int((y1 - pad_y) * height))
    right = min(width, int((x2 + pad_x) * width + 0.999))
    bottom = min(height, int((y2 + pad_y) * height + 0.999))
    if right - left < 8 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def _capabilities(
    model_loaded: bool,
    mode: str,
    advanced_statuses: dict[str, tuple[str, str]] | None = None,
    advanced_models: dict[str, AdvancedModelRuntime] | None = None,
) -> dict[str, CapabilityStatus]:
    detector = (
        "Detector custom aktif untuk tiga kelas dan sudah melewati gate."
        if model_loaded and mode == "custom"
        else "Detector fallback demo aktif, belum dianggap model custom tervalidasi."
        if model_loaded
        else "Bobot detector belum tersedia."
    )
    unavailable = "Belum tersedia karena belum ada model dan dataset berlabel yang tervalidasi."
    capabilities = {
        "detector": CapabilityStatus(status="available" if model_loaded else "unavailable", detail=detector),
        "maturity": CapabilityStatus(status="unavailable", detail=unavailable),
        "rot": CapabilityStatus(status="unavailable", detail=unavailable),
        "disease": CapabilityStatus(status="unavailable", detail=unavailable),
        "grade": CapabilityStatus(status="unavailable", detail=unavailable),
    }
    if advanced_models is not None:
        for task, runtime in advanced_models.items():
            if task not in capabilities:
                continue
            if runtime.status == "eligible" and runtime.runnable:
                capabilities[task] = CapabilityStatus(status="available", detail=runtime.detail)
            elif runtime.status == "eligible":
                capabilities[task] = CapabilityStatus(
                    status="unavailable",
                    detail="Gate lolos, tetapi model tidak mencantumkan komoditas target.",
                )
            elif runtime.status in {"rejected", "blocked"}:
                capabilities[task] = CapabilityStatus(status=runtime.status, detail=runtime.detail)
            else:
                capabilities[task] = CapabilityStatus(status="unavailable", detail=runtime.detail)
    for task, (status, detail) in (advanced_statuses or {}).items():
        if task not in capabilities:
            continue
        if status == "eligible":
            capabilities[task] = CapabilityStatus(
                status="unavailable",
                detail=f"{detail} Runtime inference lanjutan belum terpasang.",
            )
        elif status in {"rejected", "blocked"}:
            capabilities[task] = CapabilityStatus(status=status, detail=detail)
    return capabilities


class PredictionService:
    def __init__(
        self,
        provider: Callable[[Image.Image], list[RawDetection]] | None,
        model_loaded: bool,
        mode: str,
        threshold: float = 0.45,
        advanced_statuses: dict[str, tuple[str, str]] | None = None,
        advanced_models: dict[str, AdvancedModelRuntime] | None = None,
    ) -> None:
        self.provider = provider
        self.model_loaded = model_loaded
        self.mode = mode
        self.threshold = threshold
        self.advanced_statuses = advanced_statuses or {}
        self.advanced_models = advanced_models or {}

    def _assess_conditions(self, image: Image.Image, box: tuple[float, float, float, float], fruit: str) -> dict[str, ConditionAssessment]:
        result: dict[str, ConditionAssessment] = {}
        for task in ("maturity", "rot", "disease", "grade"):
            runtime = self.advanced_models.get(task)
            if runtime is None or not runtime.runnable or fruit not in runtime.fruits:
                continue
            crop = _crop_detection(image, box)
            if crop is None:
                continue
            try:
                prediction = runtime.predict(crop)
            except Exception as exc:  # runtime model tidak boleh menjatuhkan request
                result[task] = ConditionAssessment(
                    task=task,
                    status="review",
                    detail=f"Model gagal dijalankan ({type(exc).__name__}).",
                )
                continue
            display = CONDITION_DISPLAY[task].get(prediction.label)
            if display is None:
                result[task] = ConditionAssessment(
                    task=task,
                    status="review",
                    detail="Model mengembalikan label di luar taxonomy.",
                )
                continue
            if prediction.confidence < CONDITION_MIN_CONFIDENCE[task] or prediction.margin < CONDITION_MIN_MARGIN[task]:
                result[task] = ConditionAssessment(
                    task=task,
                    status="review",
                    confidence=round(prediction.confidence, 4),
                    detail="Keyakinan model rendah; ambil foto ulang dengan objek lebih jelas.",
                )
                continue
            result[task] = ConditionAssessment(
                task=task,
                status="available",
                label=prediction.label,
                label_display=display,
                confidence=round(prediction.confidence, 4),
            )
        return result

    def _limitation(self) -> str:
        active = [task for task, runtime in self.advanced_models.items() if task in {"maturity", "rot", "disease", "grade"} and runtime.runnable]
        if active:
            fruits = sorted({fruit for task in active for fruit in self.advanced_models[task].fruits})
            return (
                f"Penilaian kondisi per objek aktif untuk: {', '.join(active)} "
                f"(komoditas: {', '.join(fruits) if fruits else 'tidak ada'}). "
                "Hasil hanya berlaku pada domain dataset sumber dan tetap ditandai review bila keyakinan rendah."
            )
        return "Kematangan, busuk, penyakit, ukuran fisik, dan grade belum tersedia tanpa model khusus dan data berlabel."

    def predict(self, image: Image.Image) -> PredictionResponse:
        original = image.convert("RGB")
        enhanced, enhancement = enhance_image(original)
        limitation = self._limitation()
        if not self.model_loaded or self.provider is None:
            return PredictionResponse(
                model_available=False,
                mode="unavailable",
                detections=[],
                message="Model terlatih belum tersedia. Tambahkan bobot model lalu ulangi.",
                limitation=limitation,
                original_image=_image_data_url(original),
                enhanced_image=_image_data_url(enhanced),
                enhancement_status=enhancement.status,
                enhancement_steps=enhancement.applied_steps,
                quality=_report_to_schema(enhancement),
                capabilities=_capabilities(False, self.mode, self.advanced_statuses, self.advanced_models),
                recommendations=_recommendations(enhancement),
                ai=None,
            )

        detections: list[Detection] = []
        assessed = 0
        for raw in self.provider(enhanced):
            label, raw_stage = split_label(raw.label, raw.stage)
            if (
                not isfinite(raw.confidence)
                or raw.confidence < (DEMO_REVIEW_THRESHOLD if self.mode == "demo" else self.threshold)
                or raw.confidence > 1.0
                or label not in FRUIT_NAMES
                or (raw_stage is not None and raw_stage not in STAGE_NAMES)
            ):
                continue
            x1, y1, x2, y2 = raw.box
            if not all(isfinite(value) for value in raw.box):
                continue
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                continue
            stage = STAGE_NAMES.get((raw_stage or "").lower())
            box_area = (x2 - x1) * (y2 - y1)
            warnings: list[str] = []
            if box_area < 0.01:
                warnings.append("objek kecil")
            if x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99:
                warnings.append("objek menyentuh tepi")
            if x1 <= 0.03 and y1 <= 0.03 and x2 >= 0.97 and y2 >= 0.97:
                warnings.append("kemungkinan crop atau objek terpotong")
            conditions = self._assess_conditions(enhanced, (x1, y1, x2, y2), label)
            if conditions:
                assessed += 1
            needs_review = raw.confidence < max(self.threshold + 0.1, 0.6) or bool(warnings) or any(
                condition.status == "review" for condition in conditions.values()
            )
            detections.append(
                Detection(
                    id=len(detections) + 1,
                    class_name=label,
                    fruit=FRUIT_NAMES[label],
                    confidence=round(raw.confidence, 4),
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    stage=stage,
                    stage_available=stage is not None,
                    review_status="review" if needs_review else "accepted",
                    warnings=warnings,
                    conditions=conditions,
                )
            )
        if detections:
            message = f"{len(detections)} objek terdeteksi."
            if assessed:
                assessed_fruits = sorted({item.fruit.lower() for item in detections if item.conditions})
                fruit_text = ", ".join(assessed_fruits)
                message += f" {assessed} objek dinilai kondisi {fruit_text}."
        else:
            message = "Tidak ada objek dari tiga kelas yang dikenali."
        return PredictionResponse(
            model_available=True,
            mode=self.mode if self.mode in {"custom", "demo", "test"} else "demo",
            detections=detections,
            message=message,
            limitation=limitation,
            original_image=_image_data_url(original),
            enhanced_image=_image_data_url(enhanced),
            enhancement_status=enhancement.status,
            enhancement_steps=enhancement.applied_steps,
            quality=_report_to_schema(enhancement),
            capabilities=_capabilities(True, self.mode, self.advanced_statuses, self.advanced_models),
            recommendations=_recommendations(enhancement),
            ai=None,
        )


def _recommendations(report: EnhancementReport) -> list[str]:
    recommendations: list[str] = []
    if "gelap" in report.before.warnings:
        recommendations.append("Gunakan cahaya dari samping atau depan objek, bukan membelakangi sumber cahaya.")
    if "terlalu_terang" in report.before.warnings or "silau" in report.before.warnings:
        recommendations.append("Kurangi pantulan cahaya dan ambil foto dari sudut sedikit berbeda.")
    if "noise_tinggi" in report.before.warnings:
        recommendations.append("Gunakan tempat yang lebih terang agar kamera tidak menaikkan noise.")
    if not recommendations:
        recommendations.append("Pastikan objek terlihat utuh, tidak tertutup, dan memenuhi bagian tengah foto.")
    return recommendations
