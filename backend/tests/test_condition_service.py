from PIL import Image

from app.inference.base import RawDetection
from app.services.advanced_model_registry import ConditionPrediction
from app.services.prediction_service import PredictionService


class StubRuntime:
    status = "eligible"
    runnable = True
    detail = "Model stub aktif untuk tomat."
    fruits = ("tomato",)

    def __init__(self, prediction: ConditionPrediction, task: str = "maturity") -> None:
        self.task = task
        self._prediction = prediction

    def predict(self, crop: Image.Image) -> ConditionPrediction:
        return self._prediction


def _service(runtime: StubRuntime) -> PredictionService:
    return PredictionService(
        provider=lambda _image: [
            RawDetection(label="tomato", confidence=0.9, box=(0.1, 0.1, 0.5, 0.5)),
            RawDetection(label="cucumber", confidence=0.9, box=(0.6, 0.1, 0.9, 0.4)),
        ],
        model_loaded=True,
        mode="test",
        advanced_models={"maturity": runtime},
    )


def test_condition_is_applied_only_to_matching_fruit() -> None:
    result = _service(StubRuntime(ConditionPrediction("ripe", 0.92, 0.55))).predict(
        Image.new("RGB", (200, 200), "white")
    )

    tomato = result.detections[0]
    cucumber = result.detections[1]

    assert tomato.conditions["maturity"].status == "available"
    assert tomato.conditions["maturity"].label == "ripe"
    assert tomato.conditions["maturity"].confidence == 0.92
    assert cucumber.conditions == {}
    assert result.capabilities["maturity"].status == "available"


def test_low_confidence_condition_becomes_review_without_label() -> None:
    result = _service(StubRuntime(ConditionPrediction("ripe", 0.51, 0.02))).predict(
        Image.new("RGB", (200, 200), "white")
    )

    condition = result.detections[0].conditions["maturity"]

    assert condition.status == "review"
    assert condition.label is None
    assert condition.label_display is None
    assert result.detections[0].review_status == "review"
