from pathlib import Path
from PIL import Image
import pytest

from app.config import load_settings
from app.services.model_registry import load_runtime
from app.services.prediction_service import PredictionService, split_label, PRODUCTION_FRUIT_NAMES


def test_production_fruit_names_contains_35_classes():
    assert "apple" in PRODUCTION_FRUIT_NAMES
    assert "banana" in PRODUCTION_FRUIT_NAMES
    assert "orange" in PRODUCTION_FRUIT_NAMES
    assert "tomato" in PRODUCTION_FRUIT_NAMES
    assert "strawberry" in PRODUCTION_FRUIT_NAMES
    assert "watermelon" in PRODUCTION_FRUIT_NAMES
    assert len(PRODUCTION_FRUIT_NAMES) >= 35


def test_split_label_recognizes_new_fruits():
    fruit, stage = split_label("Apple", None)
    assert fruit == "apple"
    assert stage is None

    fruit, stage = split_label("strawberry_ripe", None)
    assert fruit == "strawberry"
    assert stage == "ripe"


def test_load_runtime_with_fruit_veg_model():
    model_path = Path("models/fruit_veg_yolo11m.pt")
    if not model_path.is_file():
        pytest.skip("models/fruit_veg_yolo11m.pt tidak ada di lokal")

    runtime = load_runtime(model_path, confidence_threshold=0.25)
    assert runtime.model_loaded is True
    assert runtime.mode == "custom"
    assert len(runtime.class_names) == 35
    assert "apple" in runtime.class_names or "Apple" in runtime.class_names or "tomato" in runtime.class_names


def test_prediction_service_localizes_classes():
    service = PredictionService(
        provider=None,
        model_loaded=False,
        mode="custom",
        class_names=("apple", "banana", "watermelon"),
    )
    assert service.fruit_names["apple"] == "Apel"
    assert service.fruit_names["banana"] == "Pisang"
    assert service.fruit_names["watermelon"] == "Semangka"
