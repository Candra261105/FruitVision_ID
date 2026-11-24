from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


class Box(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_order(self) -> "Box":
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError("Bounding box harus memiliki luas positif")
        return self


class ConditionAssessment(BaseModel):
    task: Literal["maturity", "rot", "disease", "grade"]
    status: Literal["available", "review"]
    label: str | None = None
    label_display: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str = ""


class ColorIndicator(BaseModel):
    hue_segments: dict[str, float]
    saturation: float = Field(ge=0.0, le=1.0)
    value: float = Field(ge=0.0, le=1.0)
    status: Literal["available", "review"]
    detail: str = "Indikator warna HSV, bukan diagnosis kondisi buah."


class Detection(BaseModel):
    id: int = Field(ge=1)
    class_name: str
    fruit: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: Box
    stage: str | None = None
    stage_available: bool = False
    review_status: Literal["accepted", "review"] = "accepted"
    warnings: list[str] = Field(default_factory=list)
    conditions: dict[str, ConditionAssessment] = Field(default_factory=dict)
    crop_image: str | None = None
    color_indicator: ColorIndicator | None = None


class QualityReport(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    brightness: float = Field(ge=0.0, le=255.0)
    contrast: float = Field(ge=0.0, le=255.0)
    noise: float = Field(ge=0.0)
    glare_ratio: float = Field(ge=0.0, le=1.0)
    blur_score: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class CapabilityStatus(BaseModel):
    status: Literal["available", "unavailable", "review_required", "rejected", "blocked"]
    detail: str


class AIExplanation(BaseModel):
    status: Literal["available", "unavailable", "error"]
    detail: str
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    disclaimer: str


class AIExplainRequest(BaseModel):
    inspection_id: Annotated[str, StringConstraints(min_length=32, max_length=32)]


class PredictionResponse(BaseModel):
    model_available: bool
    mode: Literal["custom", "demo", "test", "unavailable"]
    detections: list[Detection]
    message: str
    limitation: str
    original_image: str | None = None
    enhanced_image: str | None = None
    enhancement_status: Literal["enhanced", "unchanged"] = "unchanged"
    enhancement_steps: list[str] = Field(default_factory=list)
    quality: QualityReport | None = None
    capabilities: dict[str, CapabilityStatus] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    ai: AIExplanation | None = None
    inspection_id: str | None = None


class ModelInfo(BaseModel):
    name: str
    mode: str
    model_available: bool
    classes: list[str]
    stage_classes: list[str]
    limitation: str
