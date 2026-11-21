from __future__ import annotations

from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError
from starlette.concurrency import run_in_threadpool

from app.schemas.prediction import AIExplainRequest, AIExplanation, ModelInfo, PredictionResponse
from app.services.ai_explainer import AIRequest
from app.services.model_registry import model_info
from app.services.prediction_service import PredictionService

router = APIRouter()


def _service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    runtime = request.app.state.runtime
    return {"status": "ok", "model_loaded": runtime.model_loaded, "mode": runtime.mode}


@router.get("/api/model-info", response_model=ModelInfo)
def get_model_info(request: Request) -> ModelInfo:
    return ModelInfo.model_validate(model_info(request.app.state.runtime))


@router.get("/api/calibration-status")
def calibration_status() -> dict[str, object]:
    return {
        "available": False,
        "message": "Estimasi hari menuju matang belum dikalibrasi dengan data waktu yang valid.",
    }


@router.post("/api/explain", response_model=AIExplanation)
async def explain_prediction(request: Request, payload: AIExplainRequest) -> AIExplanation:
    client_key = request.client.host if request.client else "unknown"
    if not request.app.state.ai_rate_limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Batas permintaan AI per menit tercapai. Coba lagi nanti.")
    ai_request = request.app.state.inspections.pop(payload.inspection_id, None)
    if ai_request is None:
        raise HTTPException(status_code=404, detail="Hasil inspeksi tidak ditemukan atau sudah digunakan.")
    if not ai_request.detections and not ai_request.quality_warnings:
        raise HTTPException(status_code=422, detail="Hasil inspeksi belum memiliki bukti CV untuk dijelaskan.")
    if not request.app.state.settings.ai_enabled:
        result = request.app.state.ai_explainer.unavailable("AI provider belum diaktifkan di backend.")
    else:
        async with request.app.state.ai_semaphore:
            result = await run_in_threadpool(
                request.app.state.ai_explainer.explain,
                ai_request,
            )
    return AIExplanation.model_validate(result.__dict__)


@router.post("/api/predict/image", response_model=PredictionResponse)
async def predict_image(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> PredictionResponse:
    settings = request.app.state.settings
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Ukuran gambar terlalu besar. Batasnya "
                f"{settings.max_upload_bytes / (1024 * 1024):.1f} MB."
            ),
        )
    try:
        with Image.open(BytesIO(content)) as opened:
            width, height = opened.size
            if (
                width <= 0
                or height <= 0
                or width > settings.max_image_dimension
                or height > settings.max_image_dimension
                or width * height > settings.max_image_pixels
            ):
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Dimensi atau jumlah pixel gambar melewati batas keamanan.",
                )
            image = opened.convert("RGB")
    except HTTPException:
        raise
    except (DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="File yang dikirim bukan gambar yang dapat dibaca.",
        ) from exc
    async with request.app.state.inference_semaphore:
        result = await run_in_threadpool(_service(request).predict, image)
    import uuid

    inspection_id = uuid.uuid4().hex
    request.app.state.inspections[inspection_id] = AIRequest(
        detections=[
            {
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "review_status": detection.review_status,
            }
            for detection in result.detections
        ],
        quality_warnings=result.quality.warnings if result.quality else [],
    )
    return result.model_copy(update={"inspection_id": inspection_id})
