from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    confidence_threshold: float
    max_upload_bytes: int
    max_image_pixels: int
    max_image_dimension: int
    max_concurrent_inferences: int
    cors_origins: tuple[str, ...]
    ai_enabled: bool
    ai_max_concurrent: int
    ai_max_requests_per_minute: int
    allow_unapproved_custom: bool


def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    configured_model = os.getenv("FRUITVISION_MODEL_PATH", "")
    model_path = Path(configured_model) if configured_model else root / "models" / "yolov8s-world.pt"
    if not model_path.is_absolute():
        model_path = root / model_path
    origins = os.getenv("FRUITVISION_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return Settings(
        model_path=model_path,
        confidence_threshold=float(os.getenv("FRUITVISION_CONFIDENCE", "0.45")),
        max_upload_bytes=int(os.getenv("FRUITVISION_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
        max_image_pixels=int(os.getenv("FRUITVISION_MAX_IMAGE_PIXELS", str(16_777_216))),
        max_image_dimension=int(os.getenv("FRUITVISION_MAX_IMAGE_DIMENSION", "8192")),
        max_concurrent_inferences=int(os.getenv("FRUITVISION_MAX_CONCURRENT_INFERENCES", "1")),
        cors_origins=tuple(item.strip() for item in origins.split(",") if item.strip()),
        ai_enabled=os.getenv("FRUITVISION_AI_ENABLED", "false").lower() == "true",
        ai_max_concurrent=int(os.getenv("FRUITVISION_AI_MAX_CONCURRENT", "1")),
        ai_max_requests_per_minute=int(os.getenv("FRUITVISION_AI_MAX_REQUESTS_PER_MINUTE", "10")),
        allow_unapproved_custom=os.getenv("FRUITVISION_ALLOW_UNAPPROVED_CUSTOM", "false").lower() == "true",
    )
