from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import load_settings
from app.services.model_registry import load_runtime
from app.services.ai_explainer import AIExplainer
from app.services.rate_limit import SlidingWindowRateLimiter
from app.services.prediction_service import PredictionService
from app.services.advanced_model_registry import load_advanced_models


def create_app() -> FastAPI:
    settings = load_settings()
    runtime = load_runtime(
        settings.model_path,
        settings.confidence_threshold,
        allow_unapproved_custom=settings.allow_unapproved_custom,
    )
    advanced_models = load_advanced_models(settings.model_path.parent)
    service = PredictionService(
        provider=runtime.provider,
        model_loaded=runtime.model_loaded,
        mode=runtime.mode,
        threshold=settings.confidence_threshold,
        advanced_models=advanced_models,
    )
    application = FastAPI(title="FruitVision ID", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.state.settings = settings
    application.state.runtime = runtime
    application.state.advanced_models = advanced_models
    application.state.prediction_service = service
    application.state.ai_explainer = AIExplainer()
    application.state.ai_semaphore = asyncio.Semaphore(settings.ai_max_concurrent)
    application.state.inspections = {}
    application.state.ai_rate_limiter = SlidingWindowRateLimiter(settings.ai_max_requests_per_minute)
    application.state.inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
    application.include_router(router)
    return application


app = create_app()
