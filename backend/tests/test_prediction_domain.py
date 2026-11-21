from PIL import Image

from app.inference.base import RawDetection
from app.services.prediction_service import PredictionService


def test_normalize_keeps_multiple_objects() -> None:
    service = PredictionService(
        provider=lambda _image: [
            RawDetection(label="cucumber", confidence=0.91, box=(0.1, 0.2, 0.3, 0.4), stage="ripe"),
            RawDetection(label="tomato", confidence=0.88, box=(0.31, 0.2, 0.5, 0.4), stage="unripe"),
        ],
        model_loaded=True,
        mode="test",
    )

    result = service.predict(Image.new("RGB", (100, 100), "white"))

    assert len(result.detections) == 2
    assert [item.id for item in result.detections] == [1, 2]
    assert result.detections[0].fruit == "Mentimun"
    assert result.detections[1].stage == "mentah"


def test_unavailable_provider_returns_explicit_status() -> None:
    service = PredictionService(provider=None, model_loaded=False, mode="unavailable")

    result = service.predict(Image.new("RGB", (100, 100), "white"))

    assert result.detections == []
    assert result.model_available is False
    assert "belum tersedia" in result.message.lower()


def test_low_confidence_and_unknown_labels_are_rejected() -> None:
    service = PredictionService(
        provider=lambda _image: [
            RawDetection(label="cucumber", confidence=0.44, box=(0, 0, 1, 1)),
            RawDetection(label="dragonfruit", confidence=0.99, box=(0, 0, 1, 1)),
        ],
        model_loaded=True,
        mode="test",
    )

    result = service.predict(Image.new("RGB", (20, 20), "white"))

    assert result.detections == []


def test_demo_low_confidence_detection_is_visible_as_review() -> None:
    service = PredictionService(
        provider=lambda _image: [
            RawDetection(label="tomato", confidence=0.24, box=(0.0, 0.0, 1.0, 1.0)),
        ],
        model_loaded=True,
        mode="demo",
        threshold=0.45,
    )

    result = service.predict(Image.new("RGB", (160, 160), "white"))

    assert len(result.detections) == 1
    assert result.detections[0].review_status == "review"
    assert any("tepi" in warning for warning in result.detections[0].warnings)


def test_detection_that_fills_frame_is_marked_as_possible_crop() -> None:
    service = PredictionService(
        provider=lambda _image: [
            RawDetection(label="tomato", confidence=0.91, box=(0.02, 0.02, 0.98, 0.98)),
        ],
        model_loaded=True,
        mode="custom",
        threshold=0.45,
    )

    result = service.predict(Image.new("RGB", (640, 360), "white"))

    assert result.detections[0].review_status == "review"
    assert any("crop" in warning for warning in result.detections[0].warnings)


def test_combined_fruit_and_stage_label_is_normalized() -> None:
    service = PredictionService(
        provider=lambda _image: [
            RawDetection(label="cucumber_ripe", confidence=0.9, box=(0, 0, 1, 1)),
        ],
        model_loaded=True,
        mode="test",
    )

    result = service.predict(Image.new("RGB", (20, 20), "white"))

    assert result.detections[0].class_name == "cucumber"
    assert result.detections[0].stage == "siap"
    assert result.detections[0].stage_available is True


def test_unknown_stage_and_invalid_box_are_rejected() -> None:
    service = PredictionService(
        provider=lambda _image: [
            RawDetection(label="cucumber_unknown", confidence=0.9, box=(0, 0, 1, 1)),
            RawDetection(label="cucumber", confidence=0.9, box=(0.8, 0.2, 0.2, 0.9)),
            RawDetection(label="tomato", confidence=0.9, box=(0, 0, 0, 0.5)),
        ],
        model_loaded=True,
        mode="test",
    )

    result = service.predict(Image.new("RGB", (20, 20), "white"))

    assert result.detections == []


def test_prediction_includes_capability_status_for_unavailable_analyses() -> None:
    service = PredictionService(provider=None, model_loaded=False, mode="unavailable")

    result = service.predict(Image.new("RGB", (20, 20), "white"))

    assert result.capabilities["disease"].status == "unavailable"
    assert result.capabilities["maturity"].status == "unavailable"
    assert result.quality is not None


def test_prediction_keeps_advanced_capabilities_independent() -> None:
    service = PredictionService(
        provider=None,
        model_loaded=False,
        mode="unavailable",
        advanced_statuses={"maturity": ("eligible", "Model maturity lolos gate.")},
    )

    result = service.predict(Image.new("RGB", (20, 20), "white"))

    assert result.capabilities["maturity"].status == "unavailable"
    assert "runtime" in result.capabilities["maturity"].detail.lower()
    assert result.capabilities["disease"].status == "unavailable"


def test_prediction_message_names_detected_condition_fruit() -> None:
    from app.services.advanced_model_registry import AdvancedModelRuntime

    disease = AdvancedModelRuntime(
        "disease",
        "eligible",
        model_path=__import__("pathlib").Path("disease.pt"),
        detail="aktif",
        labels=("healthy", "belly_rot", "pythium_fruit_rot"),
        fruits=("cucumber",),
    )
    disease.predict = lambda _image: type("Prediction", (), {"label": "belly_rot", "confidence": 0.95, "margin": 0.5})()
    service = PredictionService(
        provider=lambda _image: [RawDetection(label="cucumber", confidence=0.9, box=(0.1, 0.1, 0.8, 0.8))],
        model_loaded=True,
        mode="test",
        advanced_models={"disease": disease},
    )

    result = service.predict(Image.new("RGB", (100, 100), "white"))

    assert "kondisi mentimun" in result.message.lower()
