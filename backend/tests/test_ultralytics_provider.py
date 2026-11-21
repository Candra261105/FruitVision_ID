from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from app.inference.ultralytics_provider import UltralyticsProvider
from app.services.model_registry import load_runtime


def test_open_vocabulary_provider_sets_target_classes(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeWorldModel:
        def __init__(self, model_path: str) -> None:
            calls["model_path"] = model_path
            self.names = {}

        def set_classes(self, classes: list[str]) -> None:
            calls["classes"] = classes
            self.names = dict(enumerate(classes))

        def predict(self, **_kwargs):
            class Values:
                def __init__(self, values) -> None:
                    self.values = values

                def tolist(self):
                    return self.values

            boxes = SimpleNamespace(
                xyxy=Values([[2.0, 4.0, 18.0, 20.0]]),
                conf=Values([0.91]),
                cls=Values([0]),
            )
            return [SimpleNamespace(boxes=boxes, names={0: "banana"})]

    monkeypatch.setitem(
        __import__("sys").modules,
        "ultralytics",
        SimpleNamespace(YOLOWorld=FakeWorldModel),
    )

    provider = UltralyticsProvider(
        Path("models/yolov8s-world.pt"),
        confidence_threshold=0.45,
        class_prompts=("banana", "mango", "papaya", "avocado", "orange"),
    )

    detections = provider(Image.new("RGB", (20, 20), "white"))

    assert Path(str(calls["model_path"])) == Path("models/yolov8s-world.pt")
    assert calls["classes"] == ["banana", "mango", "papaya", "avocado", "orange"]
    assert provider.is_open_vocabulary is True
    assert provider.class_names == (
        "banana",
        "mango",
        "papaya",
        "avocado",
        "orange",
    )
    assert detections[0].label == "banana"


def test_registry_accepts_open_vocabulary_model_without_static_class_names(
    monkeypatch, tmp_path
) -> None:
    class FakeWorldProvider:
        is_open_vocabulary = True
        class_names = ()

        def __init__(self, _model_path, _confidence_threshold, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "app.inference.ultralytics_provider.UltralyticsProvider",
        FakeWorldProvider,
        raising=False,
    )
    model_path = tmp_path / "yolov8s-world.pt"
    model_path.write_bytes(b"weights")

    runtime = load_runtime(model_path, confidence_threshold=0.45)

    assert runtime.model_loaded is True
    assert runtime.mode == "demo"


def test_registry_does_not_promote_unapproved_custom_model(monkeypatch, tmp_path) -> None:
    class FakeCustomProvider:
        is_open_vocabulary = False
        class_names = ("cucumber", "tomato", "green beans")

        def __init__(self, _model_path, _confidence_threshold, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "app.inference.ultralytics_provider.UltralyticsProvider",
        FakeCustomProvider,
        raising=False,
    )
    model_path = tmp_path / "fruit_detector.pt"
    model_path.write_bytes(b"weights")

    runtime = load_runtime(model_path, confidence_threshold=0.45)

    assert runtime.model_loaded is False
    assert runtime.mode == "unavailable"
    assert "gate" in runtime.limitation.lower()


def test_registry_allows_explicit_trial_custom_model(monkeypatch, tmp_path) -> None:
    class FakeCustomProvider:
        is_open_vocabulary = False
        class_names = ("cucumber", "tomato", "green beans")

        def __init__(self, _model_path, _confidence_threshold, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "app.inference.ultralytics_provider.UltralyticsProvider",
        FakeCustomProvider,
        raising=False,
    )
    model_path = tmp_path / "yolo26n-trial.pt"
    model_path.write_bytes(b"weights")

    runtime = load_runtime(model_path, confidence_threshold=0.45, allow_unapproved_custom=True)

    assert runtime.model_loaded is True
    assert runtime.mode == "trial"
    assert "uji" in runtime.limitation.lower()


def test_registry_loads_approved_custom_model_without_world_prompts(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeCustomProvider:
        is_open_vocabulary = False
        class_names = ("cucumber", "tomato", "green beans")

        def __init__(self, _model_path, _confidence_threshold, **kwargs) -> None:
            calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "app.inference.ultralytics_provider.UltralyticsProvider",
        FakeCustomProvider,
        raising=False,
    )
    model_path = tmp_path / "fruit_detector.pt"
    model_path.write_bytes(b"weights")
    model_path.with_suffix(".gate.json").write_text(
        '{"status":"eligible","classes":["cucumber","tomato","green beans"],"imgsz":640}',
        encoding="utf-8",
    )

    runtime = load_runtime(model_path, confidence_threshold=0.45)

    assert runtime.model_loaded is True
    assert runtime.mode == "custom"
    assert calls["kwargs"] == {"imgsz": 640}


def test_open_vocabulary_provider_retries_small_crop_with_context(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeWorldModel:
        def __init__(self, _model_path: str) -> None:
            self.names = {0: "tomato"}

        def set_classes(self, _classes: list[str]) -> None:
            return None

        def predict(self, **kwargs):
            calls.append(kwargs)
            if kwargs["conf"] >= 0.45:
                return [SimpleNamespace(boxes=None, names={0: "tomato"})]

            class Values:
                def __init__(self, values) -> None:
                    self.values = values

                def tolist(self):
                    return self.values

            boxes = SimpleNamespace(
                xyxy=Values([[300.0, 300.0, 340.0, 340.0]]),
                conf=Values([0.24]),
                cls=Values([0]),
            )
            return [SimpleNamespace(boxes=boxes, names={0: "tomato"})]

    monkeypatch.setitem(
        __import__("sys").modules,
        "ultralytics",
        SimpleNamespace(YOLOWorld=FakeWorldModel),
    )

    provider = UltralyticsProvider(
        Path("models/yolov8s-world.pt"),
        confidence_threshold=0.45,
        class_prompts=("cucumber", "tomato", "green beans"),
    )

    detections = provider(Image.new("RGB", (160, 160), "white"))

    assert detections[0].label == "tomato"
    assert detections[0].confidence == 0.24
    assert len(calls) >= 2
    assert any(call["conf"] < 0.45 for call in calls)


def test_open_vocabulary_provider_retries_large_image_at_lower_confidence(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeWorldModel:
        def __init__(self, _model_path: str) -> None:
            self.names = {0: "tomato"}

        def set_classes(self, _classes: list[str]) -> None:
            return None

        def predict(self, **kwargs):
            calls.append(kwargs)
            if kwargs["conf"] >= 0.45:
                return [SimpleNamespace(boxes=None, names={0: "tomato"})]

            class Values:
                def __init__(self, values) -> None:
                    self.values = values

                def tolist(self):
                    return self.values

            return [SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=Values([[20.0, 20.0, 80.0, 80.0]]),
                    conf=Values([0.31]),
                    cls=Values([0]),
                ),
                names={0: "tomato"},
            )]

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", SimpleNamespace(YOLOWorld=FakeWorldModel))
    provider = UltralyticsProvider(
        Path("models/yolov8s-world.pt"),
        confidence_threshold=0.45,
        class_prompts=("cucumber", "tomato", "green beans"),
    )

    detections = provider(Image.new("RGB", (640, 360), "white"))

    assert detections[0].label == "tomato"
    assert detections[0].confidence == 0.31
    assert any(call["conf"] < 0.45 for call in calls)
    assert any(call["conf"] < 0.45 and call["source"].size == (640, 360) for call in calls)


def test_open_vocabulary_provider_maps_review_tile_boxes_to_original_image(monkeypatch) -> None:
    calls: list[object] = []

    class FakeWorldModel:
        def __init__(self, _model_path: str) -> None:
            self.names = {0: "tomato"}

        def set_classes(self, _classes: list[str]) -> None:
            return None

        def predict(self, **kwargs):
            calls.append(kwargs["source"])
            if len(calls) < 3:
                return [SimpleNamespace(boxes=None, names={0: "tomato"})]

            class Values:
                def __init__(self, values) -> None:
                    self.values = values

                def tolist(self):
                    return self.values

            return [SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=Values([[20.0, 20.0, 80.0, 80.0]]),
                    conf=Values([0.24]),
                    cls=Values([0]),
                ),
                names={0: "tomato"},
            )]

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", SimpleNamespace(YOLOWorld=FakeWorldModel))
    provider = UltralyticsProvider(
        Path("models/yolov8s-world.pt"),
        confidence_threshold=0.45,
        class_prompts=("cucumber", "tomato", "green beans"),
    )

    detections = provider(Image.new("RGB", (640, 360), "white"))

    assert detections[0].label == "tomato"
    assert detections[0].box == (20 / 640, 20 / 360, 80 / 640, 80 / 360)
    assert len(calls) >= 3
